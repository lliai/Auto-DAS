import unittest
from unittest import TestCase

from diswotv2.api.api import DisWOTAPI


class TestDisWOTAPI(TestCase):

    def setUp(self) -> None:
        self.api = DisWOTAPI(
            path='./data/diswotv2_dict_4e1ea4a046d7e970b9151d8e23332ec5.pkl')

    def test_random_struct(self):
        struct = self.api.random_struct()
        print(struct)

    def test_query_by_struct(self):
        struct = self.api.random_struct()
        acc = self.api.query_by_struct(struct)
        print(struct, acc)

    def test_next(self):
        struct, acc = next(self.api)
        print(struct, acc)

        struct, acc = next(self.api)
        print(struct, acc)

    def test_iter(self):
        for i, j in iter(self.api):
            print(i, j)


if __name__ == '__main__':
    unittest.main()
