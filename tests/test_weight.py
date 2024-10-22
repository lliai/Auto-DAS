import unittest
from unittest import TestCase

import torch

import diswotv2.primitives.weight  # noqa: F401
from diswotv2.primitives import build_weight


class TestWeight(TestCase):

    def setUp(self) -> None:
        self.f = torch.tensor(1.0)
        self.t = torch.randn(3, 4, 5, 6)
        self.s = torch.randn(3, 4, 5, 6)

    def test_w1(self):
        w1 = build_weight('w1_teacher')
        o = w1(self.f, self.t)
        print(o.shape, o)

    def test_w1_ts(self):
        w1_ts = build_weight('w1_teacher_student')
        o = w1_ts(self.f, self.t, self.s)
        print(o.shape, o)

    def test_w5(self):
        w5 = build_weight('w5_teacher')
        o = w5(self.f, self.t)
        print(o.shape, o)

    def test_w5_ts(self):
        w5_ts = build_weight('w5_teacher_student')
        o = w5_ts(self.f, self.t, self.s)
        print(o.shape, o)

    def test_w25(self):
        w25 = build_weight('w25_teacher')
        o = w25(self.f, self.t)
        print(o.shape, o)

    def test_w25_ts(self):
        w25_ts = build_weight('w25_teacher_student')
        o = w25_ts(self.f, self.t, self.s)
        print(o.shape, o)

    def test_w50(self):
        w50 = build_weight('w50_teacher')
        o = w50(self.f, self.t)
        print(o.shape, o)

    def test_w50_ts(self):
        w50_ts = build_weight('w50_teacher_student')
        o = w50_ts(self.f, self.t, self.s)
        print(o.shape, o)

    def test_w100(self):
        w100 = build_weight('w100_teacher')
        o = w100(self.f, self.t)
        print(o.shape, o)

    def test_w100_ts(self):
        w100_ts = build_weight('w100_teacher_student')
        o = w100_ts(self.f, self.t, self.s)
        print(o.shape, o)


if __name__ == '__main__':
    unittest.main()
