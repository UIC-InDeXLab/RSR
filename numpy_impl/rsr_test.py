import unittest

import numpy as np

from multipliers_np import (
    RSRBinaryMultiplier,
    NaiveMultiplier,
    RSRTernaryMultiplier,
    RSRPlusPlusBinaryMultiplier,
    RSRPlusPlusTernaryMultiplier,
)


class TestRSRMultiplier(unittest.TestCase):
    @staticmethod
    def generate_random_int_vector(size, low=0, high=100):
            random_vector = np.random.randint(low, high, size)
            return random_vector
    
    def setUp(self):
        self.n = 2 ** 12
        self.m = 2 ** 13 + 5
        self.v = TestRSRMultiplier.generate_random_int_vector(self.n)

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

    def test_rsr_pp_binary_multiplier(self):
        def generate_random_binary_matrix(n):
            binary_matrix = np.random.randint(2, size=(n, n))
            return binary_matrix

        A = generate_random_binary_matrix(self.n)
        expected_result = NaiveMultiplier(A).multiply(self.v)
        rsr_pp_multiplier = RSRPlusPlusBinaryMultiplier(A)
        rsr_pp_result = rsr_pp_multiplier.multiply(self.v)

        np.testing.assert_allclose(rsr_pp_result, expected_result, rtol=1e-6, atol=1e-6)

    def test_rsr_pp_ternary_multiplier(self):
        def generate_random_ternary_matrix(n):
            ternary_matrix = np.random.randint(low=-1, high=2, size=(n, n))
            return ternary_matrix

        A = generate_random_ternary_matrix(self.n)
        expected_result = NaiveMultiplier(A).multiply(self.v)
        rsr_pp_multiplier = RSRPlusPlusTernaryMultiplier(A)
        rsr_pp_result = rsr_pp_multiplier.multiply(self.v)

        np.testing.assert_allclose(rsr_pp_result, expected_result, rtol=1e-6, atol=1e-6)

    def test_rsr_binary_non_square(self):
        def generate_random_binary_non_square_matrix(n, m):
            binary_matrix = np.random.randint(2, size=(n, m))
            return binary_matrix

        A = generate_random_binary_non_square_matrix(self.n, self.m)
        expected_result = NaiveMultiplier(A).multiply(self.v)
        rsr_multiplier = RSRBinaryMultiplier(A)
        rsr_result = rsr_multiplier.multiply(self.v)

        np.testing.assert_allclose(rsr_result, expected_result, rtol=1e-6, atol=1e-6)

    def test_rsr_ternary_non_square(self):
        def generate_random_ternary_non_square_matrix(n, m):
            binary_matrix = np.random.randint(low=-1, high=2, size=(n, m))
            return binary_matrix

        A = generate_random_ternary_non_square_matrix(self.n, self.m)
        expected_result = NaiveMultiplier(A).multiply(self.v)
        rsr_multiplier = RSRTernaryMultiplier(A)
        rsr_result = rsr_multiplier.multiply(self.v)

        np.testing.assert_allclose(rsr_result, expected_result, rtol=1e-6, atol=1e-6)

    def test_rsrpp_binary_non_square(self):
        def generate_random_binary_non_square_matrix(n, m):
            binary_matrix = np.random.randint(2, size=(n, m))
            return binary_matrix

        A = generate_random_binary_non_square_matrix(self.n, self.m)
        expected_result = NaiveMultiplier(A).multiply(self.v)
        rsr_multiplier = RSRPlusPlusBinaryMultiplier(A)
        rsr_result = rsr_multiplier.multiply(self.v)

        np.testing.assert_allclose(rsr_result, expected_result, rtol=1e-6, atol=1e-6)

    def test_rsrpp_ternary_non_square(self):
        def generate_random_ternary_non_square_matrix(n, m):
            binary_matrix = np.random.randint(low=-1, high=2, size=(n, m))
            return binary_matrix

        A = generate_random_ternary_non_square_matrix(self.n, self.m)
        expected_result = NaiveMultiplier(A).multiply(self.v)
        rsr_multiplier = RSRPlusPlusTernaryMultiplier(A)
        rsr_result = rsr_multiplier.multiply(self.v)

        np.testing.assert_allclose(rsr_result, expected_result, rtol=1e-6, atol=1e-6)