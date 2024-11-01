import unittest

import numpy as np

from multipliers import RSRBinaryMultiplier, NaiveMultiplier, RSRTernaryMultiplier


class TestRSRMultiplier(unittest.TestCase):
    def setUp(self):
        def generate_random_int_vector(size, low=0, high=100):
            random_vector = np.random.randint(low, high, size)
            return random_vector

        self.n = 2 ** 12
        self.v = generate_random_int_vector(self.n)

    def test_rsr_multiplier(self):
        def generate_random_binary_matrix(n):
            binary_matrix = np.random.randint(2, size=(n, n))
            return binary_matrix

        A = generate_random_binary_matrix(self.n)
        expected_result = NaiveMultiplier(A).multiply(self.v)
        rsr_multiplier = RSRBinaryMultiplier(A)
        rsr_result = rsr_multiplier.multiply(self.v)

        np.testing.assert_allclose(rsr_result, expected_result, rtol=1e-6, atol=1e-6)

    def test_rsr_ternary_multiplier(self):
        def generate_random_ternary_matrix(n):
            ternary_matrix = np.random.randint(low=-1, high=2, size=(n, n))
            return ternary_matrix

        A = generate_random_ternary_matrix(self.n)
        expected_result = NaiveMultiplier(A).multiply(self.v)
        rsr_multiplier = RSRTernaryMultiplier(A)
        rsr_result = rsr_multiplier.multiply(self.v)

        np.testing.assert_allclose(rsr_result, expected_result, rtol=1e-6, atol=1e-6)
