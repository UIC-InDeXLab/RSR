import math
from collections import defaultdict
from itertools import product

from accelerate import init_empty_weights

import torch
import torch.nn as nn
import torch.nn.functional as F



def bucket_sort_rows(matrix: torch.Tensor) -> list:
    """Sort rows into buckets based on binary row patterns (0/1) efficiently."""
    n, k = matrix.shape
    # Create a dictionary to store indices for each binary pattern
    bucket = defaultdict(list)

    # Generate all possible binary row combinations for length k
    for row_tuple in product([0, 1], repeat=k):
        bucket[row_tuple] = []  # Initialize empty list for each pattern

    # Place each row index into its corresponding bucket
    for i in range(n):
        row_tuple = tuple(matrix[i].tolist())
        bucket[row_tuple].append(i)

    # Collect sorted row indices from the buckets
    sorted_indices = []
    for row_tuple in sorted(bucket):
        sorted_indices.append(tuple(bucket[row_tuple]))

    return sorted_indices


def preprocess(matrix: torch.Tensor, k: int) -> tuple:
    """Preprocess matrix to ensure divisibility by k and apply bucket sort on submatrices."""
    n, m = matrix.shape  # Get the dimensions of the input matrix

    # Check if the number of columns is divisible by k
    additional_columns = (k - (m % k)) % k  # Columns to add for divisibility by k

    # Create a new padded matrix on GPU
    if additional_columns > 0:
        padding = torch.zeros((n, additional_columns), device=matrix.device, dtype=matrix.dtype)
        matrix = torch.cat([matrix, padding], dim=1)

    # List to store sorted indices for each k-sized submatrix
    sorted_indices = []

    # Iterate over k-sized submatrices and apply bucket sort
    for split_idx in range(0, matrix.shape[1], k):
        submatrix = matrix[:, split_idx:split_idx + k]
        indices = bucket_sort_rows(submatrix)
        sorted_indices.append(indices)

    return sorted_indices, additional_columns


def generate_binary_matrix(k: int, device='cuda') -> torch.Tensor:
    """Generate all binary row combinations of length k using PyTorch tensors."""
    # Generate all combinations using tensor operations
    rows = torch.cartesian_prod(*[torch.tensor([0, 1], device=device, dtype=torch.bfloat16) for _ in range(k)])
    return rows  # Shape: (2^k, k)


# the weights are ternary so can be represented with 2 bits, and they are packed in uint8 tensors, hence the number of values per item is 4
VALUES_PER_ITEM = 4


def pack_weights(quantized_weights: torch.Tensor) -> torch.Tensor:
    """
    Packs a tensor of quantized weights into a compact format using 2 bits per value.

    Parameters:
    -----------
    quantized_weights : torch.Tensor
        A tensor containing ternary quantized weights with values in {-1, 0, 1}. These values are adjusted to
        {0, 1, 2} before being packed.

    Returns:
    --------
    torch.Tensor
        A packed tensor where each element stores 4 quantized values (each using 2 bits) in an 8-bit format.
    """

    original_shape = quantized_weights.shape

    row_dim = (original_shape[0] + VALUES_PER_ITEM - 1) // VALUES_PER_ITEM

    if len(original_shape) == 1:
        packed_tensor_shape = (row_dim,)
    else:
        packed_tensor_shape = (row_dim, *original_shape[1:])

    quantized_weights += 1
    packed = torch.zeros(packed_tensor_shape, device=quantized_weights.device, dtype=torch.uint8)
    unpacked = quantized_weights.to(torch.uint8)

    it = min(VALUES_PER_ITEM, (original_shape[0] // row_dim) + 1)
    for i in range(it):
        start = i * row_dim
        end = min(start + row_dim, original_shape[0])
        packed[: (end - start)] |= unpacked[start:end] << 2 * i

    return packed


def unpack_weights_deprecated(packed: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """
    Unpacks a tensor of quantized weights that were stored in a packed format using 2 bits per value.

    Parameters:
    -----------
    packed : torch.Tensor
        A tensor containing packed weights where each element represents 4 quantized values (using 2 bits per value).
    dtype : torch.dtype
        The dtype of the returned Tensor
    Returns:
    --------
    torch.Tensor
        A tensor of unpacked weights, where each value is converted from its packed 2-bit representation.

    Example:
    --------
    packed = torch.tensor([[0b10100001, 0b00011000],
                           [0b10010000, 0b00001010]], dtype=torch.uint8)

    # Unpack the values
    unpacked = unpack_weights(packed)

    # Resulting unpacked tensor
    print(unpacked)
    # Output: tensor([[ 0, -1],
                      [-1,  1],
                      [-1,  1],
                      [-1,  1],
                      [ 1,  0],
                      [ 0, -1],
                      [ 1, -1],
                      [ 1, -1]])

    Explanation of the example:
    ---------------------------
    Let's take the first value for example 0b10100001, we we will only focus on the first column,
    because every element is unpacked across the first dimension
    - First 2 bits: `01` → 0 at [0][0]
    - Second 2 bits: `00` → -1 at [0][2]
    - Third 2 bits: `10` → 1 at [0][4]
    - Fourth 2 bits: `10` → 1 at [0][6]
    the second value of the same row (0b10010000) will give the values for [0][1], [0][3], [0][5], [0][7]

    We subtract 1 because during the packing process, it's easier to work with values like 0, 1, and 2. To make this possible,
    we add 1 to the original ternary weights (which are typically -1, 0, and 1) when packing them. When unpacking, we reverse
    this by subtracting 1 to restore the original ternary values.
    """
    packed_shape = packed.shape

    if len(packed_shape) == 1:
        original_row_dim = packed_shape[0] * VALUES_PER_ITEM
        unpacked_shape = (original_row_dim,)
    else:
        original_row_dim = packed_shape[0] * VALUES_PER_ITEM
        unpacked_shape = (original_row_dim, *packed_shape[1:])

    unpacked = torch.zeros(unpacked_shape, device=packed.device, dtype=torch.uint8)

    for i in range(VALUES_PER_ITEM):
        start = i * packed_shape[0]
        end = start + packed_shape[0]
        mask = 3 << (2 * i)
        unpacked[start:end] = (packed & mask) >> (2 * i)

    return unpacked.to(dtype) - 1


def unpack_weights(packed: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Unpacks a tensor of quantized weights into two tensors A and B, containing 0s and 1s.
    A - B will give the original unpacked tensor (in ternary -1, 0, 1 values).
    """
    packed_shape = packed.shape

    # Determine the shape of the unpacked tensors
    if len(packed_shape) == 1:
        original_row_dim = packed_shape[0] * VALUES_PER_ITEM
        unpacked_shape = (original_row_dim,)
    else:
        original_row_dim = packed_shape[0] * VALUES_PER_ITEM
        unpacked_shape = (original_row_dim, *packed_shape[1:])

    # Create tensors A and B filled with zeros
    A = torch.zeros(unpacked_shape, device=packed.device, dtype=torch.uint8)
    B = torch.zeros(unpacked_shape, device=packed.device, dtype=torch.uint8)

    # Extract the 2-bit values from the packed tensor and assign them to A and B
    for i in range(VALUES_PER_ITEM):
        start = i * packed_shape[0]
        end = start + packed_shape[0]
        mask = 3 << (2 * i)

        # Extract the 2-bit value and assign to A and B
        value = (packed & mask) >> (2 * i)
        A[start:end] = (value == 2).to(torch.uint8)  # Set A to 1 if value is 2 or 3
        B[start:end] = (value == 0).to(torch.uint8)  # Set B to 1 if value is 1 or 3

    return A.to(dtype), B.to(dtype)


def smart_multiplication(input: torch.Tensor, k: int, permutations: list, additional_columns: int) -> torch.Tensor:
    """Perform efficient smart multiplication with GPU tensors."""
    results = []

    # Generate the binary matrix only once
    binary_matrix = generate_binary_matrix(k)

    # Loop through permutations to compute the result for each
    for perm in permutations:
        # Sum the values of v at the specified indices
        # new_v[idx] = torch.stack([v[list(t)].sum(dim=0)])
        new_v = torch.cat([input[:, :, list(t)].sum(dim=2, keepdim=True) for t in perm], dim=2)

        # new_v[idx] = v[list(t)].sum()

        # Multiply the new vector with the binary matrix and collect results
        # result = new_v.to(dtype=self.dtype) @ binary_matrix
        result = F.linear(new_v, binary_matrix.T)
        results.append(result)

    # Concatenate all results and trim additional columns
    final_result = torch.cat(results, dim=2)
    if additional_columns > 0:
        return final_result[..., :-additional_columns]
    return final_result


if __name__ == "__main__":
    input_tensor = torch.randint(-10, 10, (1, 2, 4), dtype=torch.bfloat16, device='cuda')
    weight_tensor = torch.randint(-1, 2, (4, 4), dtype=torch.bfloat16, device='cuda')

    packed_weights = pack_weights(weight_tensor.to(torch.int8))

    A, B = unpack_weights(packed_weights, dtype=torch.bfloat16)
    w_quant_mahdi = A - B
    w_quant_deprecated = unpack_weights_deprecated(packed_weights, dtype=torch.bfloat16)

    assert torch.equal(w_quant_mahdi, w_quant_deprecated), "Mismatch between unpack methods!"
    A = A.T
    B = B.T
    k = 2

    permutations_A, additional_columns_A = preprocess(A, k)
    permutations_B, additional_columns_B = preprocess(B, k)

    y_A = smart_multiplication(input_tensor, k, permutations_A, additional_columns_A)
    y_B = smart_multiplication(input_tensor, k, permutations_B, additional_columns_B)
    y_smart = y_A - y_B

    y_linear = F.linear(input_tensor, w_quant_deprecated)

    assert torch.equal(y_smart, y_linear), "Mismatch between smart multiplication and F.linear!"



