import unittest

import torch

from diswotv2.primitives.operations.binary_ops import (add, kl_loss, l1_loss,
                                                       l2_loss,
                                                       matrix_multiplication,
                                                       multiply, subtract)


class TestBinary(unittest.TestCase):

    def test_add(self):
        A = torch.rand(3, 4)
        B = torch.rand(3, 4)
        self.assertTrue(torch.allclose(add(A, B), A + B))

    def test_subtract(self):
        A = torch.rand(3, 4)
        B = torch.rand(3, 4)
        self.assertTrue(torch.allclose(subtract(A, B), A - B))

    def test_multiply(self):
        A = torch.rand(3, 4)
        B = torch.rand(3, 4)
        self.assertTrue(torch.allclose(multiply(A, B), A * B))

    def test_matrix_multiplication(self):
        A = torch.rand(3, 4)
        B = torch.rand(4, 5)
        self.assertTrue(torch.allclose(matrix_multiplication(A, B), A @ B))

    def test_l1_loss(self):
        A = torch.rand(3, 4)
        B = torch.rand(3, 4)
        self.assertTrue(
            torch.allclose(l1_loss(A, B), torch.nn.functional.l1_loss(A, B)))

    def test_l2_loss(self):
        A = torch.rand(3, 4)
        B = torch.rand(3, 4)
        self.assertTrue(
            torch.allclose(l2_loss(A, B), torch.nn.functional.mse_loss(A, B)))

    def test_kl_loss(self):
        A = torch.rand(3, 4)
        B = torch.rand(3, 4)
        self.assertTrue(
            torch.allclose(
                kl_loss(A, B),
                torch.nn.functional.kl_div(A, B, reduction='batchmean')))
