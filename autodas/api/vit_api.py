# create a thorough api like nas-bench-201
import os
import random

import torch


class DisWOT_API_VIT:
    """DisWOT Benchmark API for autoformer search space for query the struct and acc"""

    def __init__(self, path: str, mode='kd', dataset='c100', verbose=True):
        assert os.path.exists(path)
        assert mode in {'kd', 'cls'}, f'Invalid mode {mode}'
        self.verbose = verbose
        self.dataset = dataset  # c100, flower, chaoyang

        self.mode = mode
        self.path = path

        self._dict = torch.load(path)

        if self.verbose:
            print(f'Loaded {path}')

    def random_index(self):
        """random sample struct"""
        return random.choice(range(len(self._dict)))

    def query_by_index(self, index: int, dataset: str):
        """query the acc by index"""
        return self._dict[index][dataset][self.mode]

    def __next__(self):
        """return a random struct and its acc"""
        rnd_index = self.random_index()
        struct = self._dict[rnd_index]['arch']
        acc = self.query_by_index(rnd_index, dataset=self.dataset)
        if self.verbose:
            print(f'DiwWOT: index: {rnd_index} struct:{struct} acc:{acc}')
        return struct, acc

    def __repr__(self) -> str:
        """return the repr of the DisWOTAPI"""
        return f'DisWOTAPI(mode={self.mode})'

    def __iter__(self):
        """make the api iterable"""
        for item in self._dict:
            yield item['arch'], item[self.dataset][self.mode]


if __name__ == '__main__':
    # pass
    api = DisWOT_API_VIT('../data/diswotv2_autoformer.pth')
    res = []
    for struct, acc in iter(api):
        res.append((struct, acc))
        print('struct:', struct)
        print('acc:', acc)
    print(len(res))
