import unittest

import numpy as np

from multipliers import RSRMultiplier, NaiveMultiplier


class TestRSRMultiplier(unittest.TestCase):

    def test_rsr_multiplier(self):
        def generate_random_binary_matrix(n):
            binary_matrix = np.random.randint(2, size=(n, n))
            return binary_matrix

        def generate_random_int_vector(size, low=0, high=100):
            random_vector = np.random.randint(low, high, size)
            return random_vector
        # Define a small matrix and vector for testing
        A = generate_random_binary_matrix(2**10)
        v = generate_random_int_vector(2 ** 10)

        # Calculate the expected result using np.dot
        expected_result = NaiveMultiplier(A).multiply(v)

        # Instantiate the RSRMultiplier with the matrix
        rsr_multiplier = RSRMultiplier(A)

        # Get the result from RSRMultiplier
        rsr_result = rsr_multiplier.multiply(v)

        # Assert that the RSR result matches the expected result from np.dot
        np.testing.assert_allclose(rsr_result, expected_result, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
