import os
import pickle
import random

import nasbench
from nasbench import api

from diswotv2.models.nasbench101.model import Network as NBNetwork

nasbench_path = '/home/stack/project/PicoNAS/data/benchmark/nasbench_only108.tfrecord'
nb = api.NASBench(nasbench_path)


def get_nb101_model(_hash):
    m = nb.get_metrics_from_hash(_hash)
    ops = m[0]['module_operations']
    adjacency = m[0]['module_adjacency']
    return NBNetwork((adjacency, ops))


def query_nb101_acc(_hash):
    m = nb.get_metrics_from_hash(_hash)
    return m[1][108][0]['final_test_accuracy']


def get_nb101_teacher():
    return get_nb101_model('b148c5e43b9b1be81b8245c2bc0c1cf3')


def get_nb101_model_and_acc(_hash):
    return get_nb101_model(_hash), query_nb101_acc(_hash)


def get_rnd_nb101_and_acc():
    ava_hashs = nb.hash_iterator()
    rand_hash = random.sample(ava_hashs, 1)[0]
    return get_nb101_model(rand_hash), query_nb101_acc(rand_hash), rand_hash


class NB101API:
    """NB101 Benchmark API for S0 search space for query the hash and acc

    FEATURE:
        support original NB101 search space.
        support results under distillation.
    """

    def __init__(self, path: str, verbose=True):
        assert os.path.exists(path), f'Invalid path {path}'
        self.verbose = verbose

        self.nb101_dict = pickle.load(open(path, 'rb'))
        self.hash_list = list(self.nb101_dict.keys())

    def random_hash(self):
        """random sample hash"""
        return random.choice(self.hash_list)

    def query_by_hash(self, hash: str):
        """query the acc by hash"""
        return float(self.nb101_dict[hash])

    def __next__(self):
        """return a random hash and its acc"""
        hash = self.random_hash()
        acc = float(self.nb101_dict[hash])
        if self.verbose:
            print(f'NB101: hash:{hash} acc:{acc}')
        return hash, acc

    def __repr__(self) -> str:
        """return the repr of the NB101API"""
        return f'NB101API(paht={self.path})'

    def __iter__(self):
        """make the api iterable"""
        yield from self.nb101_dict.items()

    def get_nb101_model(self, _hash):
        m = nb.get_metrics_from_hash(_hash)
        ops = m[0]['module_operations']
        adjacency = m[0]['module_adjacency']
        return NBNetwork((adjacency, ops))
