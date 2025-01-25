import unittest

import torch

from multipliers import (
    RSRBinaryMultiplier,
    NaiveMultiplier,
    RSRTernaryMultiplier,
    RSRPlusPlusTernaryMultiplier,
    RSRPlusPlusBinaryMultiplier
)

DTYPE = torch.float32
PRECISION = 1e-6

class TestRSRMultiplier(unittest.TestCase):    
    @staticmethod
    def generate_random_int_vector(size, low=0, high=100):
            return torch.randint(low, high, (1, size)).to(dtype=DTYPE)
    
    def setUp(self):
        self.n = 100
        self.m = 300
        self.v = TestRSRMultiplier.generate_random_int_vector(self.n)
    
    def generate_random_binary_matrix(self, n, m):
        return torch.randint(0, 2, (n, m)).to(dtype=DTYPE)
    
    def generate_random_ternary_matrix(self, n, m):
        return torch.randint(-1, 2, (n, m)).to(dtype=DTYPE)

    def test_rsr_multiplier(self):
        A = self.generate_random_binary_matrix(self.n, self.n)

        expected_result = NaiveMultiplier(A).multiply(self.v)
        rsr_multiplier = RSRBinaryMultiplier(A)
        rsr_result = rsr_multiplier.multiply(self.v)

        assert torch.allclose(rsr_result, expected_result, rtol=PRECISION, atol=PRECISION)

    def test_rsr_ternary_multiplier(self):
        A = self.generate_random_ternary_matrix(self.n, self.n)
        expected_result = NaiveMultiplier(A).multiply(self.v)
        rsr_multiplier = RSRTernaryMultiplier(A)
        rsr_result = rsr_multiplier.multiply(self.v)

        assert torch.allclose(rsr_result, expected_result, rtol=PRECISION, atol=PRECISION)

    # def test_rsr_pp_binary_multiplier(self):
    #     A = self.generate_random_binary_matrix(self.n, self.n)
    #     expected_result = NaiveMultiplier(A).multiply(self.v)
    #     rsr_pp_multiplier = RSRPlusPlusBinaryMultiplier(A)
    #     rsr_pp_result = rsr_pp_multiplier.multiply(self.v)

    #     assert torch.allclose(rsr_pp_result, expected_result, rtol=PRECISION, atol=PRECISION)

    # def test_rsr_pp_ternary_multiplier(self):
    #     A = self.generate_random_ternary_matrix(self.n, self.n)
    #     expected_result = NaiveMultiplier(A).multiply(self.v)
    #     rsr_pp_multiplier = RSRPlusPlusTernaryMultiplier(A)
    #     rsr_pp_result = rsr_pp_multiplier.multiply(self.v)

    #     assert torch.allclose(rsr_pp_result, expected_result, rtol=PRECISION, atol=PRECISION)

    def test_rsr_binary_non_square(self):
        A = self.generate_random_binary_matrix(self.n, self.m)
        expected_result = NaiveMultiplier(A).multiply(self.v)
        rsr_multiplier = RSRBinaryMultiplier(A)
        rsr_result = rsr_multiplier.multiply(self.v)

        assert torch.allclose(rsr_result, expected_result, rtol=PRECISION, atol=PRECISION)        

    def test_rsr_ternary_non_square(self):
        A = self.generate_random_ternary_matrix(self.n, self.m)
        expected_result = NaiveMultiplier(A).multiply(self.v)
        rsr_multiplier = RSRTernaryMultiplier(A)
        rsr_result = rsr_multiplier.multiply(self.v)
        print(rsr_result.shape, expected_result.shape)

        assert torch.allclose(rsr_result, expected_result, rtol=PRECISION, atol=PRECISION)        

    # def test_rsrpp_binary_non_square(self):
    #     A = self.generate_random_binary_matrix(self.n, self.m)
    #     expected_result = NaiveMultiplier(A).multiply(self.v)
    #     rsr_multiplier = RSRPlusPlusBinaryMultiplier(A)
    #     rsr_result = rsr_multiplier.multiply(self.v)

    #     assert torch.allclose(rsr_result, expected_result, rtol=PRECISION, atol=PRECISION)        

    # def test_rsrpp_ternary_non_square(self):
    #     A = self.generate_random_ternary_matrix(self.n, self.m)
    #     expected_result = NaiveMultiplier(A).multiply(self.v)
    #     rsr_multiplier = RSRPlusPlusTernaryMultiplier(A)
    #     rsr_result = rsr_multiplier.multiply(self.v)

    #     assert torch.allclose(rsr_result, expected_result, rtol=PRECISION, atol=PRECISION)        

    def test_dump_load(self):
        A = self.generate_random_ternary_matrix(100, 200)
        rsr_multiplier = RSRTernaryMultiplier(A)
        rsr_multiplier_2 = RSRTernaryMultiplier.deserialize(rsr_multiplier.serialize())

        b11 = rsr_multiplier._rsr_B1._blocks_permutations
        b12 = rsr_multiplier_2._rsr_B1._blocks_permutations
        b21 = rsr_multiplier._rsr_B2._blocks_permutations
        b22 = rsr_multiplier_2._rsr_B2._blocks_permutations
        assert len(b11) == len(b12)
        assert torch.allclose(b11[0][0], b12[0][0])
        assert torch.allclose(b11[1][1], b12[1][1])
        assert rsr_multiplier._rsr_B1.k == rsr_multiplier_2._rsr_B1.k
        assert rsr_multiplier._rsr_B1.m == rsr_multiplier_2._rsr_B1.m

        assert torch.allclose(b21[0][0], b22[0][0])
        assert torch.allclose(b21[1][1], b22[1][1])
        assert len(b21) == len(b22)
        assert rsr_multiplier._rsr_B2.k == rsr_multiplier_2._rsr_B2.k
        assert rsr_multiplier._rsr_B2.m == rsr_multiplier_2._rsr_B2.m