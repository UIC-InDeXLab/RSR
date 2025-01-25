import math
from abc import abstractmethod, ABC
from typing import Type

import numpy as np
from pympler import asizeof

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
    Dump the only necessary `data` to run the `multiply()`
    '''
    @abstractmethod
    def dump(self):
        pass

    '''
    Given the output of `dump()`, create a minimal object
    '''
    @classmethod
    def load(cls, data):
        pass


class NaiveMultiplier(MatrixMultiplier):
    def multiply(self, v):
        return np.dot(v, self.A)
    
    def dump(self):
        return self.A
    
    @classmethod
    def load(cls, data):
        return cls(A=data)


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
        sorted_indices = np.lexsort(matrix.T[::-1])
        return sorted_indices

    @staticmethod
    def find_segment_indices(matrix, permutation):
        sorted_matrix = matrix[permutation]
        n, k = sorted_matrix.shape

        segment_start_indices = np.full(2 ** k, n, dtype=int)

        current_row_str = format(0, f'0{k}b')
        current_row_decimal = 0
        last_row_decimal = None
        segment_start_indices[current_row_decimal] = 0

        for i, row in enumerate(sorted_matrix):
            row_str = ''.join(map(str, row))

            if row_str != current_row_str:
                segment_start_indices[int(row_str, 2)] = i
                last_row_decimal = current_row_decimal
                current_row_decimal = int(row_str, 2)
                current_row_str = row_str
                for r in range(last_row_decimal + 1, current_row_decimal):
                    segment_start_indices[r] = i

        return segment_start_indices

    @staticmethod
    def generate_binary_matrix(k):
        num_rows = 2 ** k
        binary_matrix = np.array([list(map(int, np.binary_repr(i, width=k))) for i in range(num_rows)])
        return binary_matrix

    def preprocess(self, k, debug=True):
        padding = (k - (self.m % k)) % k  # Padding to make column count divisible by k

        if padding > 0:
            A_padded = np.pad(self.A, ((0, 0), (0, padding)), mode='constant')
        else:
            A_padded = self.A

        blocks_permutations = []

        for split_idx in range(0, self.m, k):
            block_matrix = A_padded[:, split_idx:split_idx + k]

            permutations = self.find_permutations(block_matrix)

            segment_indices = self.find_segment_indices(block_matrix, permutations)

            blocks_permutations.append((permutations, segment_indices))

        if debug:
            total_memory = asizeof.asizeof(blocks_permutations)
            print(f"Total memory for blocks_permutations: {total_memory} bytes")

        return blocks_permutations, padding

    def multiply(self, v):
        results = np.empty(self.m + self._padding, dtype=int)
        binary_matrix = self.generate_binary_matrix(self.k)
        naive_multiplier = NaiveMultiplier(binary_matrix)

        for i, (permutation, segment_indices) in enumerate(self._blocks_permutations):
            segmented_sum = np.empty(2 ** self.k)
            permuted_vector = v[permutation]
            cumsum_arr = np.cumsum(permuted_vector)
            cumsum_arr = np.insert(cumsum_arr, 0, 0)
            for j, start in enumerate(segment_indices):
                end = segment_indices[j + 1] if j + 1 < len(segment_indices) else len(cumsum_arr) - 1
                segmented_sum[j] = cumsum_arr[end] - cumsum_arr[start]

            results[i * self.k:i * self.k + self.k] = (naive_multiplier.multiply(segmented_sum))

        return results[:-self._padding] if self._padding > 0 else results
    
    def dump(self):
        return (
            self.m, 
            self.k,
            [np.concatenate((per, seg)) for (per, seg) in self._blocks_permutations],
            self._padding
        )
    
    @classmethod
    def load(cls, data):
        m = data[0]
        k = data[1]
        block_permutation = [(row[:-2**k], row[-2**k:]) for row in data[2]]
        padding = data[3]

        obj = cls()
        obj._m = m
        obj._k = k
        obj._blocks_permutations = block_permutation
        obj._padding = padding
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
        B1 = (matrix == 1).astype(int)
        B2 = (matrix == -1).astype(int)

        return B1, B2

    def multiply(self, v):
        return self._rsr_B1.multiply(v) - self._rsr_B2.multiply(v)

    def dump(self):
        return (
            self._rsr_B1.dump(),
            self._rsr_B2.dump()
        )

    @classmethod
    def load(cls, data):
        B1 = RSRBinaryMultiplier.load(data[0])
        B2 = RSRBinaryMultiplier.load(data[1])

        obj = cls()
        obj._rsr_B1 = B1
        obj._rsr_B2 = B2
        return obj


class RSRPlusPlusBinaryMultiplier(RSRBinaryMultiplier):
    def _faster_mult(self, segmented_sum):
        result = np.empty(self.k)
        for i in range(self.k, 0, -1):
            result[i - 1] = np.sum(segmented_sum[1::2])
            segmented_sum = segmented_sum[::2] + segmented_sum[1::2]  # TODO: reuse [1::2]
        return result

    def multiply(self, v):
        results = np.empty(self.m + self._padding, dtype=int)

        for i, (permutation, segment_indices) in enumerate(self._blocks_permutations):
            segmented_sum = np.empty(2 ** self.k)
            permuted_vector = v[permutation]
            cumsum_arr = np.cumsum(permuted_vector)
            cumsum_arr = np.insert(cumsum_arr, 0, 0)
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