import os
import pickle
import random

from nas_201_api import NASBench201API

from diswotv2.models.nasbench201.utils import (dict2config,
                                               get_cell_based_tiny_net)

nb201_api = NASBench201API(
    file_path_or_dict='data/NAS-Bench-201-v1_0-e61699.pth', verbose=False)


def get_teacher_best_model(TARGET='cifar100', NUM_CLASSES=100):
    best_idx, high_accurcy = nb201_api.find_best(
        dataset=TARGET,  # ImageNet16-120
        metric_on_set='test',
        hp='200')
    arch_config = {
        'name': 'infer.tiny',
        'C': 16,
        'N': 5,
        'arch_str': nb201_api.arch(best_idx),
        'num_classes': NUM_CLASSES
    }
    net_config = dict2config(arch_config, None)
    return get_cell_based_tiny_net(net_config)


def random_sample_and_get_gt(TARGET='cifar100', NUM_CLASSES=100):
    index_range = list(range(15625))
    choiced_index = random.choice(index_range)
    # modelinfo is a index
    # modelinfo = 15624
    arch_config = {
        'name': 'infer.tiny',
        'C': 16,
        'N': 5,
        'arch_str': nb201_api.arch(choiced_index),
        'num_classes': NUM_CLASSES
    }
    net_config = dict2config(arch_config, None)
    model = get_cell_based_tiny_net(net_config)
    xinfo = nb201_api.get_more_info(choiced_index, dataset=TARGET, hp='200')
    return model, xinfo['test-accuracy']


def query_gt_by_arch_str(arch_str, TARGET='cifar100'):
    choiced_index = nb201_api.query_index_by_arch(arch_str)
    xinfo = nb201_api.get_more_info(choiced_index, dataset=TARGET, hp='200')
    return xinfo['test-accuracy']


def get_network_by_index(choiced_index):
    arch_config = {
        'name': 'infer.tiny',
        'C': 16,
        'N': 5,
        'arch_str': nb201_api.arch(choiced_index),
        'num_classes': 100
    }
    net_config = dict2config(arch_config, None)
    model = get_cell_based_tiny_net(net_config)
    xinfo = nb201_api.get_more_info(
        choiced_index, dataset='cifar100', hp='200')
    return model, xinfo['test-accuracy']


def get_network_by_archstr(arch_str):
    arch_config = {
        'name': 'infer.tiny',
        'C': 16,
        'N': 5,
        'arch_str': arch_str,
        'num_classes': 100
    }
    net_config = dict2config(arch_config, None)
    return get_cell_based_tiny_net(net_config)


class NB201KDAPI:
    """NB201 Benchmark API for S0 search space for query the idx and acc

    FEATURE:
        support original NB201 search space.
        support results under distillation.
    """

    def __init__(self, path: str, verbose=True):
        assert os.path.exists(path), f'Invalid path {path}'
        self.verbose = verbose

        self.NB201_dict = pickle.load(open(path, 'rb'))
        self.idx_list = list(self.NB201_dict.keys())

    def random_idx(self):
        """random sample idx"""
        return random.choice(self.idx_list)

    def query_by_idx(self, idx: str):
        """query the acc by idx"""
        return float(self.NB201_dict[idx])

    def __next__(self):
        """return a random idx and its acc"""
        idx = self.random_idx()
        acc = float(self.NB201_dict[idx])
        if self.verbose:
            print(f'NB201: idx:{idx} acc:{acc}')
        return idx, acc

    def __repr__(self) -> str:
        """return the repr of the NB201KDAPI"""
        return f'NB201KDAPI(paht={self.path})'

    def __iter__(self):
        """make the api iterable"""
        yield from self.NB201_dict.items()
