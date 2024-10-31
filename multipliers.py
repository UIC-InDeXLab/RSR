import math
from abc import abstractmethod, ABC

import numpy as np


class MatrixMultiplier(ABC):
    def __init__(self, A):
        self._A = A
        self._n = len(A)

    @abstractmethod
    def multiply(self, v):
        pass

    @property
    def A(self):
        return self._A

    @property
    def n(self):
        return self._n


class NaiveMultiplier(MatrixMultiplier):
    def multiply(self, v):
        return np.dot(v, self.A)


class RSRMultiplier(MatrixMultiplier):
    def __init__(self, A):
        super().__init__(A)
        self._k = int(math.log(self.n, 2) - math.log(math.log(self.n, 2), 2))
        # self._k = 2
        self._blocks_permutations, self._padding = self.preprocess(self._k)

    @property
    def k(self):
        return self._k

    @staticmethod
    def find_permutations(matrix):
        # Sort the rows lexicographically to get sorted indices
        sorted_indices = np.lexsort(matrix.T[::-1])
        return sorted_indices

    @staticmethod
    def find_segment_indices(matrix, permutation):
        # Apply the permutation to sort the matrix
        sorted_matrix = matrix[permutation]
        n, k = matrix.shape

        # Initialize a dictionary to store start and end indices for each unique row
        binary_groups = [format(i, f'0{k}b') for i in range(2 ** k)]
        segment_indices = {group: (0, 0) for group in binary_groups}

        # Initialize variables to track the current row pattern and the start of each segment
        current_row = None
        start_index = 0

        for i, row in enumerate(sorted_matrix):
            # Convert row to a string to use as a dictionary key
            row_str = ''.join(map(str, row))

            # If we encounter a new row pattern, store the previous segment
            if row_str != current_row:
                if current_row is not None:
                    segment_indices[current_row] = (start_index, i)

                # Update the current row and start index
                current_row = row_str
                start_index = i

        # Add the final segment after the loop completes
        if current_row is not None:
            segment_indices[current_row] = (start_index, len(sorted_matrix))

        return segment_indices

    @staticmethod
    def generate_binary_matrix(k):
        # Generate all binary combinations and convert them to a NumPy array
        num_rows = 2 ** k
        binary_matrix = np.array([list(map(int, np.binary_repr(i, width=k))) for i in range(num_rows)])
        return binary_matrix

    def preprocess(self, k):
        # Check if self.n is divisible by k; if not, calculate padding
        padding = (k - (self.n % k)) % k  # Padding to make divisible by k

        # Pad the matrix with zeros along columns if needed
        if padding > 0:
            A_padded = np.pad(self.A, ((0, 0), (0, padding)), mode='constant')
        else:
            A_padded = self.A

        blocks_permutations = []

        # Process each block of columns of size k
        for split_idx in range(0, A_padded.shape[1], k):
            block_matrix = A_padded[:, split_idx:split_idx + k]

            # Find permutation for sorting this block
            permutations = self.find_permutations(block_matrix)

            # Find segment indices in the sorted block matrix
            segment_indices = self.find_segment_indices(block_matrix, permutations)

            blocks_permutations.append((permutations, segment_indices))

        return blocks_permutations, padding

    def multiply(self, v):
        results = np.zeros(self.n + self._padding, dtype=int)
        binary_matrix = self.generate_binary_matrix(self.k)
        naive_multiplier = NaiveMultiplier(binary_matrix)

        for i, (permutation, segment_indices) in enumerate(self._blocks_permutations):
            segmented_sum = np.zeros(2 ** self.k)
            permuted_vector = v[permutation]
            cumsum_arr = np.cumsum(permuted_vector)
            cumsum_arr = np.insert(cumsum_arr, 0, 0)
            for group, (start, end) in segment_indices.items():
                segmented_sum[int(group, 2)] = cumsum_arr[end] - cumsum_arr[start]

            results[i * self.k:i * self.k + self.k] = (naive_multiplier.multiply(segmented_sum))

        return results[:-self._padding] if self._padding > 0 else results


class RSRPlusPlusMultiplier(RSRMultiplier):
    def multiply(self, v):
        pass
