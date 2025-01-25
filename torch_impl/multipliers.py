import math
from abc import abstractmethod, ABC
from typing import Type

import torch
from torch.nn import functional as F
from pympler import asizeof
from line_profiler import profile

'''
Calculates the multiplication of an integer vector of size n to a matrix of size (n, m) 
'''
class MatrixMultiplier(ABC):
    def __init__(self, A=None):
        self._A = A
        self._n = 0
        self._m = 0
        if A is not None:
            self._n = len(A) # rows count
            self._m = len(A[0]) # columns count

    @abstractmethod
    def multiply(self, v):
        pass

    @property
    def A(self):
        return self._A

    @property
    def n(self):
        return self._n
    
    @property
    def m(self):
        return self._m
    
    '''
    Serialize the only necessary `data` to run the `multiply()`
    You can use `torch.save(...)` to store the data.
    '''
    @abstractmethod
    def serialize(self):
        pass

    '''
    Given the output of `serialize()`, create a minimal object
    '''
    @classmethod
    def deserialize(cls, data):
        pass


class NaiveMultiplier(MatrixMultiplier):
    @profile
    def multiply(self, v):
        return F.linear(v, self.A.T).to(dtype=v.dtype) # v.A^T
    
    def serialize(self):
        return {
            "A": self.A
        }
    
    @classmethod
    def deserialize(cls, data):
        return cls(A=data["A"])


class RSRBinaryMultiplier(MatrixMultiplier):
    def __init__(self, A=None, k: int = None):
        super().__init__(A)
        if k is None and A is not None:
            self._k = int(math.log(self.n, 2) - math.log(math.log(self.n, 2), 2))
        else:
            self._k = k
        if A is not None:
            self._blocks_permutations, self._padding = self.preprocess(self.k)
        self._A = None

    @property
    def k(self):
        return self._k

    @staticmethod
    def find_permutations(matrix):
        powers_of_two = 2 ** torch.arange(matrix.size(1) - 1, -1, -1, dtype=matrix.dtype)
        row_values = matrix.matmul(powers_of_two)
        sorted_indices = torch.argsort(row_values)
        return sorted_indices.to(dtype=torch.int)

    @staticmethod
    def find_segment_indices(matrix, permutation):
        sorted_matrix = matrix[permutation]
        n, k = sorted_matrix.shape

        segment_start_indices = torch.full((2 ** k, ), n, dtype=torch.int)

        current_row_str = format(0, f'0{k}b')
        current_row_decimal = 0
        last_row_decimal = None
        segment_start_indices[current_row_decimal] = 0

        for i, row in enumerate(sorted_matrix):
            row_str = ''.join(map(lambda x: str(int(x)), row.tolist()))

            if row_str != current_row_str:
                segment_start_indices[int(row_str, 2)] = i
                last_row_decimal = current_row_decimal
                current_row_decimal = int(row_str, 2)
                current_row_str = row_str
                for r in range(last_row_decimal + 1, current_row_decimal):
                    segment_start_indices[r] = i

        return segment_start_indices

    @staticmethod
    def generate_binary_matrix(n, type):
        integers = torch.arange(2**n, dtype=torch.long)
        binary_matrix = (integers[:, None] >> torch.arange(n - 1, -1, -1, dtype=torch.long)) & 1
        return binary_matrix.to(dtype=type)

    def preprocess(self, k):
        self.binary_matrix = self.generate_binary_matrix(self.k, self.A.dtype)
        
        padding = (k - (self.m % k)) % k  # Padding to make column count divisible by k

        if padding > 0:
            A_padded = F.pad(self.A, (0, padding), mode='constant')
        else:
            A_padded = self.A

        blocks_permutations = []

        for split_idx in range(0, self.m, k):
            block_matrix = A_padded[:, split_idx:split_idx + k]
            
            binary_values = (block_matrix * (2 ** torch.arange(k - 1, -1, -1))).sum(dim=1, dtype=torch.int)

            num_columns = 2 ** k
            L = torch.zeros((len(block_matrix), num_columns), dtype=self.A.dtype)

            L[torch.arange(len(block_matrix)), binary_values] = 1.0
            
            blocks_permutations.append(L)
            
        return torch.stack(blocks_permutations), padding

    @profile
    def multiply(self, v):
        if self._padding > 0:
            return (v @ self._blocks_permutations @ self.binary_matrix).permute(1, 0, 2).reshape(v.size(0), -1)[:, :-self._padding]
        else:
            return (v @ self._blocks_permutations @ self.binary_matrix).permute(1, 0, 2).reshape(v.size(0), -1)
    
    def serialize(self):
        data = {
            "m": self.m,
            "k": self.k,
            "padding": self._padding,
            "blocks": self._blocks_permutations
        }
        return data
    
    @classmethod
    def deserialize(cls, data):
        obj = cls()
        obj._m = data["m"]
        obj._k = data["k"]
        obj._blocks_permutations = data["blocks"]
        obj._padding = data["padding"]
        return obj


class RSRTernaryMultiplier(MatrixMultiplier):
    def __init__(self, A=None, k: int = None):
        super().__init__(A)
        if A is not None:
            self._B1, self._B2 = RSRTernaryMultiplier.unpack_matrices(A)
            bin_class = self.__binary_multiplier_class__
            self._rsr_B1 = bin_class(self._B1, k=k)
            self._rsr_B2 = bin_class(self._B2, k=k)

    @property
    def __binary_multiplier_class__(self) -> Type[RSRBinaryMultiplier]:
        return RSRBinaryMultiplier

    @staticmethod
    def unpack_matrices(matrix):
        B1 = (matrix == 1).to(dtype=matrix.dtype)
        B2 = (matrix == -1).to(dtype=matrix.dtype)

        return B1, B2
    
    '''
    Use this to get a single matrix `c`. 
    The final result can be efficiently calculated as `v @ c`.
    '''
    def get_agg_matrix(self):
        b = self._rsr_B1._blocks_permutations - self._rsr_B2._blocks_permutations
        c = b @ self._rsr_B1.binary_matrix
        return c

    def multiply(self, v):
        return self._rsr_B1.multiply(v) - self._rsr_B2.multiply(v)

    def serialize(self):
        data = {
            "B1": self._rsr_B1.serialize(),
            "B2": self._rsr_B2.serialize()
        }
        return data

    @classmethod
    def deserialize(cls, data):
        B1 = RSRBinaryMultiplier.deserialize(data["B1"])
        B2 = RSRBinaryMultiplier.deserialize(data["B2"])

        obj = cls()
        obj._rsr_B1 = B1
        obj._rsr_B2 = B2
        return obj


class RSRPlusPlusBinaryMultiplier(RSRBinaryMultiplier):
    def _faster_mult(self, segmented_sum):
        # result = np.empty(self.k)
        result = torch.empty(self.k)
        for i in range(self.k, 0, -1):
            result[i - 1] = torch.sum(segmented_sum[1::2], dim=0)
            segmented_sum = segmented_sum[::2] + segmented_sum[1::2]  # TODO: reuse [1::2]
        return result

    def multiply(self, v):
        results = torch.empty(self.m + self._padding, dtype=v.dtype)

        for i, (permutation, segment_indices) in enumerate(self._blocks_permutations):
            # segmented_sum = np.empty(2 ** self.k)
            segmented_sum = torch.empty(2 ** self.k)
            permuted_vector = v[permutation]
            cumsum_arr = torch.cumsum(permuted_vector, dim=0)
            cumsum_arr = torch.cat((torch.tensor([0]), cumsum_arr))
            for j, start in enumerate(segment_indices):
                end = segment_indices[j + 1] if j + 1 < len(segment_indices) else len(cumsum_arr) - 1
                segmented_sum[j] = cumsum_arr[end] - cumsum_arr[start]

            # CHANGE
            results[i * self.k:i * self.k + self.k] = (self._faster_mult(segmented_sum))

        return results[:-self._padding] if self._padding > 0 else results


class RSRPlusPlusTernaryMultiplier(RSRTernaryMultiplier):
    @property
    def __binary_multiplier_class__(self) -> Type[RSRBinaryMultiplier]:
        return RSRPlusPlusBinaryMultiplier
