# create a thorough api like nas-bench-201
import os
import pickle
import random


class DisWOTAPI:
    """DisWOT Benchmark API for S0 search space for query the struct and acc"""

    def __init__(self, path: str, mode='kd', verbose=True):
        assert os.path.exists(path)
        assert mode in {'kd', 'cls'}, f'Invalid mode {mode}'
        self.verbose = verbose
        self.mode = mode
        self.path = path

        _dict = pickle.load(open(path, 'rb'))

        if self.verbose:
            print(f'Loaded {path}')

        if self.mode == 'cls':
            self.struct2acc = _dict['cls']
        elif self.mode == 'kd':
            self.struct2acc = _dict['kd']

        # get the list of all structs
        self.struct_list = list(self.struct2acc.keys())

    def random_struct(self):
        """random sample struct"""
        return random.choice(self.struct_list)

    def query_by_struct(self, struct: str):
        """query the acc by struct"""
        return self.struct2acc[struct]

    def __next__(self):
        """return a random struct and its acc"""
        struct = self.random_struct()
        acc = self.struct2acc[struct]
        if self.verbose:
            print(f'DiwWOT: struct:{struct} acc:{acc}')
        return self.preprocess_struct(struct), acc

    def __repr__(self) -> str:
        """return the repr of the DisWOTAPI"""
        return f'DisWOTAPI(mode={self.mode})'

    def __iter__(self):
        """make the api iterable"""
        for struct, acc in self.struct2acc.items():
            yield self.preprocess_struct(struct), acc

    def preprocess_struct(self, struct: str):
        """convert '111' to [1, 1, 1]"""
        return [int(x) for x in struct]
