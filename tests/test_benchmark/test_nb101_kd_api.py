import unittest
from unittest import TestCase

from diswotv2.api.nas101_api import NB101API

api = NB101API(
    path='./data/nb101_kd_dict_9756ff660472a567ebabe535066c0e1f.pkl')


class TestNB101KD(TestCase):

    def test_random_hash(self):
        hash = api.random_hash()
        print(hash)
        self.assertTrue(isinstance(hash, str))
        self.assertTrue(len(hash) == 32)

    def test_query_by_hash(self):
        hash = api.random_hash()
        acc = api.query_by_hash(hash)
        print(hash, acc)
        self.assertTrue(isinstance(hash, str))
        self.assertTrue(isinstance(acc, float))

    def test_next(self):
        hash, acc = next(api)
        print(hash, acc)
        self.assertTrue(isinstance(hash, str))
        self.assertTrue(isinstance(acc, float))


if __name__ == '__main__':
    unittest.main()
