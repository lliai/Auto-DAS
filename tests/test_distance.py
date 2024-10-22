import unittest
from unittest import TestCase

import torch
import torch.nn.functional as F

import diswotv2.primitives.distance  # noqa: F401
from diswotv2.primitives import build_distance


class TestDistance(TestCase):

    def setUp(self) -> None:
        # feature type is (N, C, H, W)
        self.f_s = torch.rand(5, 4, 3, 3)
        self.f_t = torch.rand(5, 4, 3, 3)
        # logits type is (N, C)
        # self.f_s = torch.rand(5, 4)
        # self.f_t = torch.rand(5, 4)
        # middle type is (N, C, M)
        # self.f_s = torch.randn(5, 4, 3)
        # self.f_t = torch.randn(5, 4, 3)

    def test_mse_loss(self):
        mse_loss = build_distance('mse_loss')
        self.assertEqual(
            mse_loss(self.f_s, self.f_t),
            torch.nn.functional.mse_loss(self.f_s, self.f_t))

    def test_l1_loss(self):
        l1_loss = build_distance('l1_loss')
        self.assertEqual(
            l1_loss(self.f_s, self.f_t),
            torch.nn.functional.l1_loss(self.f_s, self.f_t))

    def test_l2_loss(self):
        l2_loss = build_distance('l2_loss')
        self.assertEqual(
            l2_loss(self.f_s, self.f_t),
            torch.nn.functional.mse_loss(self.f_s, self.f_t))

    def test_kl_loss(self):
        kl_loss = build_distance('kl_loss')
        self.assertEqual(
            kl_loss(self.f_s, self.f_t),
            torch.nn.functional.kl_div(
                self.f_s, self.f_t, reduction='batchmean'))

    def test_kl_T(self):
        kl_T = build_distance('kl_T1')
        self.assertEqual(
            kl_T(self.f_s, self.f_t, 10),
            torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(self.f_s / 10, dim=1),
                torch.nn.functional.softmax(self.f_t / 10, dim=1),
                reduction='batchmean') * 10 * 10)

    def test_smooth_l1_loss(self):
        smooth_l1_loss = build_distance('smooth_l1_loss')
        self.assertEqual(
            smooth_l1_loss(self.f_s, self.f_t),
            torch.nn.functional.smooth_l1_loss(self.f_s, self.f_t))

    def test_cosine_similarity(self):
        cosine_similarity = build_distance('cosine_similarity')
        aa = cosine_similarity(self.f_s, self.f_t)
        bb = F.cosine_similarity(self.f_s, self.f_t, dim=1).mean()
        self.assertTrue(torch.equal(aa, bb))

    def test_pearson_correlation(self):
        pearson_correlation = build_distance('pearson_correlation')
        # print(f' * pc: {pearson_correlation(self.f_s, self.f_t)}')
        self.assertTrue(
            len(pearson_correlation(self.f_s, self.f_t).shape) == 0)

    def test_pairwise_distance(self):
        pairwise_distance = build_distance('pairwise_distance')
        a = pairwise_distance(self.f_s, self.f_t)
        self.assertTrue(len(a.shape) == 0)

    def test_subtract(self):
        subtract = build_distance('subtract')
        self.assertEqual(
            subtract(self.f_s, self.f_t), (self.f_s - self.f_t).mean())

    def test_multiply(self):
        multiply = build_distance('multiply')
        self.assertEqual(
            multiply(self.f_s, self.f_t), (self.f_s * self.f_t).mean())

    def test_matrix_multiplication(self):
        matrix_multiplication = build_distance('matrix_multiplication')
        res = matrix_multiplication(self.f_s, self.f_t)
        self.assertTrue(len(res.shape) == 0)

    def test_lesser_than(self):
        lesser_than = build_distance('lesser_than')
        self.assertEqual(
            lesser_than(self.f_s, self.f_t),
            (self.f_s < self.f_t).float().mean())


if __name__ == '__main__':
    unittest.main()
    # logit_s = torch.ones(5, 4)
    # logit_t = torch.ones(5, 4)

    # corr = torch.einsum('bx,bx->', logit_s, logit_t)
    # print(corr)
