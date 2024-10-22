import unittest
from unittest import TestCase

import torch

import diswotv2.primitives.transform  # noqa: F401
from diswotv2.primitives import build_transform


class TestTransform(TestCase):

    def setUp(self) -> None:
        # feature
        self.f = torch.rand(2, 3, 32, 32)
        # middle
        self.m = torch.rand(3, 4, 5)
        # logits
        self.l = torch.rand(2, 10)

    def test_trans_multi_scale(self):
        print('testing trans_multi_scale...')

        # test for feature
        k_list = [1, 2, 4]
        for k in k_list:
            trans_multi_scale = build_transform(f'trans_multi_scale_r{k}')
            o = trans_multi_scale(self.f)
            self.assertTrue(len(o.shape) == 4)
            print(
                f' * multi_scale: k={k}, f.shape={self.f.shape}, o.shape={o.shape}'
            )

        # test for logits
        o = trans_multi_scale(self.l)
        self.assertTrue(len(o.shape) == 2)
        self.assertTrue(o.shape == self.l.shape)

        # test for middle
        o = trans_multi_scale(self.m)
        self.assertTrue(len(o.shape) == 3)

    def test_trans_local(self):
        print('testing trans local...')

        # test for feature
        k_list = [1, 2, 4]
        for k in k_list:
            trans_local = build_transform(f'trans_local_s{k}')
            o = trans_local(self.f)
            if k == 1:
                self.assertTrue(len(o.shape) == 2)
            else:
                self.assertTrue(len(o.shape) == 4)
            print(
                f' * local: k={k}, f.shape={self.f.shape}, o.shape={o.shape}')

        # test for logits
        o = trans_local(self.l)
        self.assertTrue(len(o.shape) == 2)
        self.assertTrue(o.shape == self.l.shape)

        # test for middle
        o = trans_local(self.m)
        self.assertTrue(len(o.shape) == 3)

    def test_trans_batch(self):
        print('testing trans batch...')
        trans_batch = build_transform('trans_batch')

        # test for feature
        o = trans_batch(self.f)
        self.assertTrue(len(o.shape) == 2)

        # test for logits
        o = trans_batch(self.l)
        self.assertTrue(len(o.shape) == 2)

        # test for middle
        o = trans_batch(self.m)
        self.assertTrue(len(o.shape) == 2)

    def test_trans_channel(self):
        print('testing trans channel...')
        trans_channel = build_transform('trans_channel')

        # test for feature
        o = trans_channel(self.f)
        self.assertTrue(len(o.shape) == 3)

        # test for logits
        o = trans_channel(self.l)
        self.assertTrue(len(o.shape) == 2)

        # test for middle
        o = trans_channel(self.m)
        self.assertTrue(len(o.shape) == 3)

    def test_trans_mask(self):
        print('testing trans mask...')
        trans_mask = build_transform('trans_mask')

        # test for feature
        o = trans_mask(self.f)
        self.assertTrue(len(o.shape) == 4)

        # test for logits
        o = trans_mask(self.l)
        self.assertTrue(len(o.shape) == 2)

        # test for middle
        o = trans_mask(self.m)
        self.assertTrue(len(o.shape) == 3)

    def test_trans_satt(self):
        print('testing trans spatial attention...')
        trans_satt = build_transform('trans_satt')

        # test for feature
        o = trans_satt(self.f)
        self.assertTrue(len(o.shape) == 4)

        # test for logits
        o = trans_satt(self.l)
        self.assertTrue(len(o.shape) == 2)

        # tes for middle
        o = trans_satt(self.m)
        self.assertTrue(len(o.shape) == 3)

    def test_trans_natt(self):
        print('testing trans batch attention...')
        trans_natt = build_transform('trans_natt')

        # test for feature
        o = trans_natt(self.f)
        self.assertTrue(len(o.shape) == 4)

        # test for logits
        o = trans_natt(self.l)
        self.assertTrue(len(o.shape) == 2)

        # test for middle
        o = trans_natt(self.m)
        self.assertTrue(len(o.shape) == 3)

    def test_trans_catt(self):
        print('testing trans channel attention...')
        trans_catt = build_transform('trans_catt')

        # test for feature
        o = trans_catt(self.f)
        self.assertTrue(len(o.shape) == 4)

        # test for logits
        o = trans_catt(self.l)
        self.assertTrue(len(o.shape) == 2)

        # test for middle
        o = trans_catt(self.m)
        self.assertTrue(len(o.shape) == 3)

    def test_trans_drop(self):
        print('testing trans drop...')
        trans_drop = build_transform('trans_drop')

        # test for feature
        o = trans_drop(self.f)
        self.assertTrue(len(o.shape) == 4)

        # test for logits
        o = trans_drop(self.l)
        self.assertTrue(len(o.shape) == 2)

        # test for middle
        o = trans_drop(self.m)
        self.assertTrue(len(o.shape) == 3)

    def test_trans_nop(self):
        print('testing trans nop...')
        trans_nop = build_transform('trans_nop')

        # test for feature
        o = trans_nop(self.f)
        self.assertTrue(len(o.shape) == 4)

        # test for logits
        o = trans_nop(self.l)
        self.assertTrue(len(o.shape) == 2)

        # test for middle
        o = trans_nop(self.m)
        self.assertTrue(len(o.shape) == 3)

    def test_trans_bmm(self):
        print('testing gram matrix...')
        trans_bmm = build_transform('trans_bmm')

        # test for feature
        o = trans_bmm(self.f)
        self.assertTrue(len(o.shape) == 3)
        self.assertTrue(o.shape[2] == o.shape[1])

        # test for logits
        o = trans_bmm(self.l)
        self.assertTrue(len(o.shape) == 2)

        # test for middle
        o = trans_bmm(self.m)
        self.assertTrue(len(o.shape) == 3)

    def test_trans_mm(self):
        print('testing gram matrix...')
        trans_mm = build_transform('trans_mm')

        # test for feature
        o = trans_mm(self.f)
        self.assertTrue(len(o.shape) == 2)

        # test for logits
        o = trans_mm(self.l)
        self.assertTrue(len(o.shape) == 2)

        # test for middle
        o = trans_mm(self.m)
        self.assertTrue(len(o.shape) == 2)

    def test_trans_norm_HW(self):
        print('testing norm in HW dim ...')
        trans_norm_HW = build_transform('trans_norm_HW')

        # test for feature
        o = trans_norm_HW(self.f)
        self.assertTrue(len(o.shape) == 4)

        # test for logits
        o = trans_norm_HW(self.l)
        self.assertTrue(len(o.shape) == 2)

        # test for middle
        o = trans_norm_HW(self.m)
        self.assertTrue(len(o.shape) == 3)

    def test_trans_norm_C(self):
        print('testing norm in C dim ...')
        trans_norm_C = build_transform('trans_norm_C')

        # test for feature
        o = trans_norm_C(self.f)
        self.assertTrue(len(o.shape) == 4)

        # test for logits
        o = trans_norm_C(self.l)
        self.assertTrue(len(o.shape) == 2)

        # test for middle
        o = trans_norm_C(self.m)
        self.assertTrue(len(o.shape) == 3)

    def test_trans_norm_N(self):
        print('testing norm in N dim ...')
        trans_norm_N = build_transform('trans_norm_N')

        # test for feature
        o = trans_norm_N(self.f)
        self.assertTrue(len(o.shape) == 4)

        # test for logits
        o = trans_norm_N(self.l)
        self.assertTrue(len(o.shape) == 2)

        # test for middle
        o = trans_norm_N(self.m)
        self.assertTrue(len(o.shape) == 3)

    # sqrt
    def test_trans_sqrt(self):
        print('testing trans sqrt ...')
        trans_sqrt = build_transform('trans_sqrt')

        # test for feature
        o = trans_sqrt(self.f)
        self.assertTrue(len(o.shape) == 4)

        # test for logits
        o = trans_sqrt(self.l)
        self.assertTrue(len(o.shape) == 2)

        # test for middle
        o = trans_sqrt(self.m)
        self.assertTrue(len(o.shape) == 3)

    # log
    def test_trans_log(self):
        print('testing trans log ...')
        trans_log = build_transform('trans_log')

        # test for feature
        o = trans_log(self.f)
        self.assertTrue(len(o.shape) == 4)

        # test for logits
        o = trans_log(self.l)
        self.assertTrue(len(o.shape) == 2)

        # test for middle
        o = trans_log(self.m)
        self.assertTrue(len(o.shape) == 3)

    # square
    def test_trans_square(self):
        print('testing trans square ...')
        trans_square = build_transform('trans_pow2')

        # test for feature
        o = trans_square(self.f)
        self.assertTrue(len(o.shape) == 4)

        # test for logits
        o = trans_square(self.l)
        self.assertTrue(len(o.shape) == 2)

        # test for middle
        o = trans_square(self.m)
        self.assertTrue(len(o.shape) == 3)

    # mix_max_norm
    def test_trans_min_max_normalize(self):
        print('testing trans min max normalize...')
        trans_min_max_normalize = build_transform('trans_min_max_normalize')

        # test for feature
        o = trans_min_max_normalize(self.f)
        self.assertTrue(len(o.shape) == 4)

        # test for logits
        o = trans_min_max_normalize(self.l)
        self.assertTrue(len(o.shape) == 2)

        # test for middle
        o = trans_min_max_normalize(self.m)
        self.assertTrue(len(o.shape) == 3)

    # exp
    def test_trans_exp(self):
        print('testing trans exp ...')
        trans_exp = build_transform('trans_exp')

        # test for feature
        o = trans_exp(self.f)
        self.assertTrue(len(o.shape) == 4)

        # test for logits
        o = trans_exp(self.l)
        self.assertTrue(len(o.shape) == 2)

        # test for middle
        o = trans_exp(self.m)
        self.assertTrue(len(o.shape) == 3)

    # abs
    def test_trans_abs(self):
        print('testing trans abs ...')
        trans_abs = build_transform('trans_abs')

        # test for feature
        o = trans_abs(self.f)
        self.assertTrue(len(o.shape) == 4)

        # test for logits
        o = trans_abs(self.l)
        self.assertTrue(len(o.shape) == 2)

        # test for middle
        o = trans_abs(self.m)
        self.assertTrue(len(o.shape) == 3)

    # sigmoid
    def test_trans_sigmoid(self):
        print('testing trans sigmoid ...')
        trans_sigmoid = build_transform('trans_sigmoid')

        # test for feature
        o = trans_sigmoid(self.f)
        self.assertTrue(len(o.shape) == 4)

        # test for logits
        o = trans_sigmoid(self.l)
        self.assertTrue(len(o.shape) == 2)

        # test for middle
        o = trans_sigmoid(self.m)
        self.assertTrue(len(o.shape) == 3)

    # scale
    def test_trans_scale(self):
        print('testing trans scale to 0-1 ...')
        trans_scale = build_transform('trans_scale')

        # test for feature
        o = trans_scale(self.f)
        self.assertTrue(len(o.shape) == 4)

        # test for logits
        o = trans_scale(self.l)
        self.assertTrue(len(o.shape) == 2)

        # test for middle
        o = trans_scale(self.m)
        self.assertTrue(len(o.shape) == 3)

    # swish
    def test_trans_swish(self):
        print('testing trans swish ...')
        trans_swish = build_transform('trans_swish')

        # test for feature
        o = trans_swish(self.f)
        self.assertTrue(len(o.shape) == 4)

        # test for logits
        o = trans_swish(self.l)
        self.assertTrue(len(o.shape) == 2)

        # test for middle
        o = trans_swish(self.m)
        self.assertTrue(len(o.shape) == 3)

    # tanh
    def test_trans_tanh(self):
        print('testing trans tanh ...')
        trans_tanh = build_transform('trans_tanh')

        # test for feature
        o = trans_tanh(self.f)
        self.assertTrue(len(o.shape) == 4)

        # test for logits
        o = trans_tanh(self.l)
        self.assertTrue(len(o.shape) == 2)

        # test for middle
        o = trans_tanh(self.m)
        self.assertTrue(len(o.shape) == 3)

    # relu
    def test_trans_relu(self):
        print('testing trans relu ...')
        trans_relu = build_transform('trans_relu')

        # test for feature
        o = trans_relu(self.f)
        self.assertTrue(len(o.shape) == 4)

        # test for logits
        o = trans_relu(self.l)
        self.assertTrue(len(o.shape) == 2)

        # test for middle
        o = trans_relu(self.m)
        self.assertTrue(len(o.shape) == 3)

    # leaky_relu

    def test_trans_leaky_relu(self):
        print('testing trans leaky_relu ...')
        trans_leaky_relu = build_transform('trans_leaky_relu')

        # test for feature
        o = trans_leaky_relu(self.f)
        self.assertTrue(len(o.shape) == 4)

        # test for logits
        o = trans_leaky_relu(self.l)
        self.assertTrue(len(o.shape) == 2)

        # test for middle
        o = trans_leaky_relu(self.m)
        self.assertTrue(len(o.shape) == 3)

    # mish

    def test_trans_mish(self):
        print('testing trans mish ...')
        trans_mish = build_transform('trans_mish')

        # test for feature
        o = trans_mish(self.f)
        self.assertTrue(len(o.shape) == 4)

        # test for logits
        o = trans_mish(self.l)
        self.assertTrue(len(o.shape) == 2)

        # test for middle
        o = trans_mish(self.m)
        self.assertTrue(len(o.shape) == 3)

    def test_trans_softmax_N(self):
        print('testing trans softmax N ...')
        trans_softmax_N = build_transform('trans_softmax_N')

        # test for feature
        o = trans_softmax_N(self.f)
        self.assertTrue(len(o.shape) == 4)

        # test for logits
        o = trans_softmax_N(self.l)
        self.assertTrue(len(o.shape) == 2)

        # test for middle
        o = trans_softmax_N(self.m)
        self.assertTrue(len(o.shape) == 3)

    def test_trans_softmax_C(self):
        print('testing trans softmax N ...')
        trans_softmax_C = build_transform('trans_softmax_C')

        # test for feature
        o = trans_softmax_C(self.f)
        self.assertTrue(len(o.shape) == 4)

        # test for logits
        o = trans_softmax_C(self.l)
        self.assertTrue(len(o.shape) == 2)

        # test for middle
        o = trans_softmax_C(self.m)
        self.assertTrue(len(o.shape) == 3)

    def test_trans_softmax_HW(self):
        print('testing trans softmax N ...')
        trans_softmax_HW = build_transform('trans_softmax_HW')

        # test for feature
        o = trans_softmax_HW(self.f)
        self.assertTrue(len(o.shape) == 3)

        # test for logits
        o = trans_softmax_HW(self.l)
        self.assertTrue(len(o.shape) == 2)

        # test for middle
        o = trans_softmax_HW(self.m)
        self.assertTrue(len(o.shape) == 3)

    def test_trans_logsoftmax_N(self):
        print('testing trans logsoftmax N ...')
        trans_logsoftmax_N = build_transform('trans_logsoftmax_N')

        # test for feature
        o = trans_logsoftmax_N(self.f)
        self.assertTrue(len(o.shape) == 4)

        # test for logits
        o = trans_logsoftmax_N(self.l)
        self.assertTrue(len(o.shape) == 2)

        # test for middle
        o = trans_logsoftmax_N(self.m)
        self.assertTrue(len(o.shape) == 3)

    def test_trans_logsoftmax_C(self):
        print('testing trans logsoftmax N ...')
        trans_logsoftmax_C = build_transform('trans_logsoftmax_C')

        # test for feature
        o = trans_logsoftmax_C(self.f)
        self.assertTrue(len(o.shape) == 4)

        # test for logits
        o = trans_logsoftmax_C(self.l)
        self.assertTrue(len(o.shape) == 2)

        # test for middle
        o = trans_logsoftmax_C(self.m)
        self.assertTrue(len(o.shape) == 3)

    def test_trans_logsoftmax_HW(self):
        print('testing trans logsoftmax N ...')
        trans_logsoftmax_HW = build_transform('trans_logsoftmax_HW')

        # test for feature
        o = trans_logsoftmax_HW(self.f)
        self.assertTrue(len(o.shape) == 3)

        # test for logits
        o = trans_logsoftmax_HW(self.l)
        self.assertTrue(len(o.shape) == 2)

        # test for middle
        o = trans_logsoftmax_HW(self.m)
        self.assertTrue(len(o.shape) == 3)

    def test_trans_batchnorm(self):
        print('testing trans logsoftmax N ...')
        trans_batchnorm = build_transform('trans_batchnorm')

        # test for feature
        o = trans_batchnorm(self.f)
        self.assertTrue(len(o.shape) == 4)

        # test for logits
        o = trans_batchnorm(self.l)
        self.assertTrue(len(o.shape) == 2)

        # test for middle
        o = trans_batchnorm(self.m)
        self.assertTrue(len(o.shape) == 3)


if __name__ == '__main__':
    unittest.main()
