"""This is an implementation of a graph data structure for autoloss."""
import math
import random

import torch
import torch.nn as nn

from diswotv2.primitives.operations import (available_zc_candidates,
                                            get_zc_candidates,
                                            sample_unary_key_by_prob,
                                            unary_operation)
from diswotv2.primitives.operations.unary_ops import UNARY_KEYS
from diswotv2.searchspace.instincts.base import BaseInstinct
from diswotv2.searchspace.instincts.utils import convert_to_float


class GraphInstinct(BaseInstinct):
    """ Graph Instinct
    Build a DAG(Directed Acyclic Graph) structure

    Args:
        n_nodes: number of nodes in the DAG

    """

    def __init__(self, n_nodes=3):
        super().__init__()
        self.n_nodes = n_nodes

        self._genotype = {
            'input_geno': [],  # only one
            'op_geno': [],  # length
        }
        # init _genotype
        self.generate_genotype()

    def sample_zc_candidates(self) -> str:
        """sample one input from zc candidates"""
        total_num_zc_candidates = len(available_zc_candidates)
        idx_zc = random.choice(range(total_num_zc_candidates))
        return available_zc_candidates[idx_zc]

    def generate_genotype(self):
        """ Randomly generate a graph structure."""
        zc_name_list = [self.sample_zc_candidates() for _ in range(2)]
        dag = []
        repr_geno = ''
        repr_geno += f'INPUT:({zc_name_list[0]}, {zc_name_list[1]})UNARY:('
        for i in range(self.n_nodes):
            dag.append([])
            for j in range(i + 2):  # include 2 input nodes
                # sample unary operation
                idx = sample_unary_key_by_prob()
                # random.choice(range(len(UNARY_KEYS)))
                dag[i].append(idx)
                repr_geno += UNARY_KEYS[idx] + '|'
            repr_geno += '-> '
        repr_geno += ')'

        # update _genotype
        self._genotype['input_geno'] = zc_name_list
        self._genotype['op_geno'] = dag
        self._repr_geno = repr_geno

    def forward_dag(self, img, label, model):
        """Forward the DAG structure to get the score."""
        if torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()
        # preprocess inputs to states
        try:
            states = [
                get_zc_candidates(
                    zc_name,
                    model,
                    device=torch.device(
                        'cuda' if torch.cuda.is_available() else 'cpu'),
                    inputs=img,
                    targets=label,
                    loss_fn=nn.CrossEntropyLoss(),
                ) for zc_name in self._genotype['input_geno']
            ]

            assert len(states) == 2, 'length of states should be 2'

        except Exception as e:
            print('GOT ERROR in GRAPH STRUCTURE: ', e)
            return -1  # invalid

        try:
            for edges in self._genotype['op_geno']:
                assert len(states) == len(
                    edges), f'length of states should be {len(edges)}'
                # states[i] is list of tensor
                midst = []
                for idx, state in zip(edges, states):
                    tmp = []
                    for s in state:
                        tmp.append(unary_operation(s, idx))
                    midst.append(tmp)

                # merge N lists of tensor to one list of tensor
                res = []
                for i in range(len(midst[0])):
                    t = midst[0][i]
                    for j in range(1, len(midst)):
                        t += midst[j][i]
                    res.append(t)
                states.append(res)

            res_list = []
            for item in states[2:]:
                res = convert_to_float(item)
                if math.isnan(res) or math.isinf(res):
                    return -1  # invalid
                res_list.append(res)
        except Exception as e:
            print('GOT ERROR in GRAPH STRUCTURE: ', e)
            return -1  # invalid

        # check whether the res_list of float have inf,nan
        return sum(res_list) / len(res_list)

    def __call__(self, img, label, model):
        return self.forward_dag(img, label, model)

    def crossover(self, other):
        """cross over two graph structures and return new genotype"""
        assert isinstance(other, GraphInstinct), 'type error'
        # crossover input_geno
        input_geno = [
            self._genotype['input_geno'][0], other._genotype['input_geno'][1]
        ]

        # crossover op_geno
        op_geno = []
        for i in range(self.n_nodes):
            op_geno.append([])
            for j in range(i + 2):
                if random.choice([True, False]):
                    op_geno[i].append(self._genotype['op_geno'][i][j])
                else:
                    op_geno[i].append(other._genotype['op_geno'][i][j])

        # rephrase repr_geno
        repr_geno = ''
        repr_geno += f'INPUT:({input_geno[0]}, {input_geno[1]})UNARY:('
        for i in range(self.n_nodes):
            for j in range(i + 2):
                repr_geno += UNARY_KEYS[op_geno[i][j]] + '|'
            repr_geno += '-> '
        repr_geno += ')'
        geno = {
            'input_geno': input_geno,
            'op_geno': op_geno,
        }
        struct = GraphInstinct(self.n_nodes)
        struct.genotype = geno
        struct._repr_geno = repr_geno
        return struct

    def mutate(self):
        """return to new genotype"""
        # input_geno
        input_geno = [
            self._genotype['input_geno'][0],
            self.sample_zc_candidates()
        ]

        # op_geno
        op_geno = []
        for i in range(self.n_nodes):
            op_geno.append([])
            for j in range(i + 2):
                if random.choice([True, False]):
                    op_geno[i].append(random.choice(range(len(UNARY_KEYS))))
                else:
                    op_geno[i].append(self._genotype['op_geno'][i][j])

        # repr_geno
        repr_geno = ''
        repr_geno += f'INPUT:({input_geno[0]}, {input_geno[1]})UNARY:('
        for i in range(self.n_nodes):
            for j in range(i + 2):
                repr_geno += f'{UNARY_KEYS[op_geno[i][j]]}|'
            repr_geno += '-> '
        repr_geno += ')'
        geno = {
            'input_geno': input_geno,
            'op_geno': op_geno,
        }

        struct = GraphInstinct(self.n_nodes)
        struct.genotype = geno
        struct._repr_geno = repr_geno
        return struct
