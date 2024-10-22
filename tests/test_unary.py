import unittest

import torch
import torch.nn.functional as F

from diswotv2.primitives.operations.unary_ops import (abs, abslog, exp, log,
                                                      min_max_normalize,
                                                      normalize, relu, revert,
                                                      sqrt, square, tanh)


class TestUnary(unittest.TestCase):

    def test_log(self):
        A = torch.rand(10)
        self.assertTrue(torch.allclose(log(A), torch.log(A)))

    def test_square(self):
        A = torch.rand(10)
        self.assertTrue(torch.allclose(square(A), torch.pow(A, 2)))

    def test_revert(self):
        A = torch.rand(10)
        self.assertTrue(torch.allclose(revert(A), A * -1))

    def test_min_max_normalize(self):
        A = torch.rand(10)
        A_min, A_max = A.min(), A.max()
        self.assertTrue(
            torch.allclose(
                min_max_normalize(A), (A - A_min) / (A_max - A_min + 1e-9)))

    def test_abslog(self):
        A = torch.rand(10)
        A[A == 0] = 1
        A = torch.abs(A)
        self.assertTrue(torch.allclose(abslog(A), torch.log(A)))

    def test_abs(self):
        A = torch.rand(10)
        self.assertTrue(torch.allclose(abs(A), torch.abs(A)))

    def test_sqrt(self):
        A = torch.rand(10)
        A[A <= 0] = 0
        self.assertTrue(torch.allclose(sqrt(A), torch.sqrt(A)))

    def test_exp(self):
        A = torch.rand(10)
        self.assertTrue(torch.allclose(exp(A), torch.exp(A)))

    def test_normalize(self):
        A = torch.rand(10)
        m = torch.mean(A)
        s = torch.std(A)
        C = (A - m) / s
        C[C != C] = 0
        self.assertTrue(torch.allclose(normalize(A), C))

    def test_relu(self):
        A = torch.rand(10)
        self.assertTrue(torch.allclose(relu(A), F.relu(A)))

    def test_tanh(self):
        A = torch.rand(10)
        self.assertTrue(torch.allclose(tanh(A), torch.tanh(A)))
