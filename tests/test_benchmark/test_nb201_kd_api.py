import unittest
from unittest import TestCase

from diswotv2.api.nas201_api import NB201KDAPI

api = NB201KDAPI(
    path='./data/nb101_kd_dict_9756ff660472a567ebabe535066c0e1f.pkl')


class TestNB201KD(TestCase):

    def test_random_idx(self):
        idx = api.random_idx()
        print(idx)
        self.assertTrue(isinstance(idx, str))
        self.assertTrue(len(idx) == 32)

    def test_query_by_idx(self):
        idx = api.random_idx()
        acc = api.query_by_idx(idx)
        print(idx, acc)
        self.assertTrue(isinstance(idx, str))
        self.assertTrue(isinstance(acc, float))

    def test_next(self):
        idx, acc = next(api)
        print(idx, acc)
        self.assertTrue(isinstance(idx, str))
        self.assertTrue(isinstance(acc, float))


if __name__ == '__main__':
    unittest.main()
