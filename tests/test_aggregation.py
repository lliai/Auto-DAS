import unittest

import torch
import torch.nn.functional as F

from diswotv2.primitives.operations.unary_ops import (channel_wise_mean,
                                                      local_max_pooling,
                                                      spatial_wise_mean)


class TestUnary(unittest.TestCase):

    def test_channel_wise_mean(self):
        A = torch.rand(3, 4)
        self.assertTrue(
            torch.allclose(channel_wise_mean(A), torch.mean(A, dim=1)))

    def test_spatial_wise_mean(self):
        A = torch.rand(3, 4, 5, 6)
        self.assertTrue(
            torch.allclose(spatial_wise_mean(A), torch.mean(A, dim=(0, 2, 3))))

    def test_local_max_pooling(self):
        A = torch.rand(3, 4, 5, 6)
        self.assertTrue(
            torch.allclose(
                local_max_pooling(A),
                F.max_pool2d(A, kernel_size=3, stride=1, padding=1)))
