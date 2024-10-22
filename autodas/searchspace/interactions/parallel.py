"""This is an implementation of a para data structure for autoloss."""
import random
import re
from copy import deepcopy

import torch.nn.functional as F

import diswotv2.primitives.distance  # noqa: F401
import diswotv2.primitives.transform  # noqa: F401
import diswotv2.primitives.weight  # noqa: F401
from diswotv2.primitives import (build_distance, build_transform, build_weight,
                                 sample_distance, sample_transform,
                                 sample_weight)
from diswotv2.searchspace.interactions.base import BaseInteraction


class ParaInteraction(BaseInteraction):
    """Build para-like search space, which is a parallel structure.
    After initialization, the ParaInteraction can work as a loss function.
    """

    def __init__(self, n_nodes=3, mutable=False):
        super().__init__()
        self.n_nodes = n_nodes  # number of transform operation.

        self._alleletype = {
            'input_allele': [],  # input type k1, k2, k3
            'trans_allele': [],  # transforms
            'weight_allele': [],  # weights
            'dists_allele': [],  # distances
        }
        # init _alleletype
        self._init_alleletype()
        self._mutable = mutable

    def __repr__(self) -> str:
        return f' * ALLELE# in:{self._alleletype["input_allele"]}~\t trans:{self._alleletype["trans_allele"]}~\t weig:{self._alleletype["weight_allele"]}~\t dist:{self._alleletype["dists_allele"]}'

    def delete_unary(self, idx):
        """Delete the idx-th transform operation."""
        assert self._mutable is True, 'ParaInteraction is not mutable.'
        assert idx < len(self._alleletype['trans_allele'])
        self._alleletype['trans_allele'].pop(idx)
        self.n_nodes -= 1

    def insert_unary(self, idx, trans_op=None):
        """Insert a transform operation at idx-th position."""
        assert self._mutable is True, 'ParaInteraction is not mutable.'
        assert idx <= len(self._alleletype['trans_allele'])
        if trans_op is None:
            self._alleletype['trans_allele'].insert(idx, sample_transform())
        else:
            self._alleletype['trans_allele'].insert(idx, trans_op)
        self.n_nodes += 1

    def __eq__(self, __o: object) -> bool:
        """Override the default Equals behavior"""
        assert isinstance(__o, ParaInteraction), 'Invalid type for comparison.'
        if self._alleletype['input_allele'] != __o._alleletype['input_allele']:
            return False
        if self._alleletype['trans_allele'] != __o._alleletype['trans_allele']:
            return False
        if self._alleletype['weight_allele'] != __o._alleletype[
                'weight_allele']:
            return False
        return self._alleletype['dists_allele'] == __o._alleletype[
            'dists_allele']

    def _init_alleletype(self):
        """ Randomly generate a para structure with parallel operations."""

        # init in allele
        input_allele = random.sample(['k1', 'k2', 'k3'], 1)
        # init transform operation with n_nodes
        trans_allele = [sample_transform() for _ in range(self.n_nodes)]
        # init weight operation
        weight_allele = [sample_weight()]
        # init distance operation
        dists_allele = [sample_distance()]
        # init _alleletype
        self._alleletype['input_allele'] = input_allele
        self._alleletype['trans_allele'] = trans_allele
        self._alleletype['weight_allele'] = weight_allele
        self._alleletype['dists_allele'] = dists_allele

    def update_alleletype(self, repr_alleletype: str):
        """Convert string representation to alleletype dictionary.

        Eg: in:['k3']~       trans:['trans_drop', 'trans_multi_scale_r4', 'trans_pow2']~     weig:['w100_teacher_student']~  dist:['l1_loss']
        """
        alleletype = {
            'input_allele': [],
            'trans_allele': [],
            'weight_allele': [],
            'dists_allele': []
        }
        parts = re.findall(r'\[.*?\]', repr_alleletype)
        alleletype['input_allele'] = re.findall(r"'(.*?)'", parts[0])
        alleletype['trans_allele'] = re.findall(r"'(.*?)'", parts[1])
        alleletype['weight_allele'] = re.findall(r"'(.*?)'", parts[2])
        alleletype['dists_allele'] = re.findall(r"'(.*?)'", parts[3])
        self._alleletype = alleletype

    def __call__(self, img, label, tmodel, smodel, interpolate=False):
        # Get features
        # import pdb; pdb.set_trace()
        s_list, s_k3 = smodel(img, is_feat=True)
        if interpolate:
            img = F.interpolate(
                img, size=(32, 32), mode='bilinear', align_corners=False)
            t_list, t_k3 = tmodel(img, is_feat=True)
        else:
            t_list, t_k3 = tmodel(img, is_feat=True)

        t_k1, t_k2 = t_list[-2], t_list[-1]
        s_k1, s_k2 = s_list[-2], s_list[-1]

        try:
            # if True:
            # Select input
            tout = eval(f't_{self._alleletype["input_allele"][0]}')
            sout = eval(f's_{self._alleletype["input_allele"][0]}')

            # Apply alignment or projection to the student branch
            # TODO

            # Perform transform operation to both branch
            for trans_op in self._alleletype['trans_allele']:
                tout = build_transform(trans_op)(tout)
                sout = build_transform(trans_op)(sout)

            # Perform distance operation
            out = build_distance(self._alleletype['dists_allele'][0])(sout,
                                                                      tout)

            # Perform transform operation
            # TODO

            # Perform weight operation
            loss = build_weight(self._alleletype['weight_allele'][0])(out,
                                                                      tout,
                                                                      sout)
            if len(loss.shape) > 1:
                import pdb
                pdb.set_trace()

        except Exception as e:
            print(f'* ParaInteraction: {self} \n error: {e}')
            return -1

        return loss.cpu().detach().numpy()

    def mutate(self):
        """return a mutated alleletype"""
        para = ParaInteraction(self.n_nodes)

        input_allele = []
        if random.random() < 0.5:
            input_allele.append(random.sample(['k1', 'k2', 'k3'], 1)[0])
        para.alleletype['input_allele'] = input_allele

        # mutate trans_allele
        para.alleletype['trans_allele'] = deepcopy(
            self._alleletype['trans_allele'])
        if random.random() < 0.5:
            # sample from depth
            rnd_idx = random.sample(range(self.n_nodes), 1)[0]
            # random sample from transform operation
            rnd_uo = sample_transform()
            para._alleletype['trans_allele'][rnd_idx] = rnd_uo

        # mutate aggregate operation
        if random.random() < 0.5:
            para.alleletype['weight_allele'] = [sample_weight()]

        # mutate binary operation
        if random.random() < 0.5:
            para.alleletype['dists_allele'] = [sample_distance()]

        return para

    def cross_over(self, other):
        """Cross over two para structure and return new one"""
        para = ParaInteraction(self.n_nodes)
        # cross over input_allele
        if random.random() < 0.5:
            para.alleletype['input_allele'] = deepcopy(
                self._alleletype['input_allele'])
        else:
            para.alleletype['input_allele'] = deepcopy(
                other._alleletype['input_allele'])

        # cross over trans_allele
        para.alleletype['trans_allele'] = deepcopy(
            self._alleletype['trans_allele'])
        rnd_idx = random.sample(range(self.n_nodes), 1)[0]
        if random.random() < 0.5:
            # sample from depth
            para._alleletype['trans_allele'][rnd_idx] = deepcopy(
                other._alleletype['trans_allele'][rnd_idx])
        else:
            para._alleletype['trans_allele'][rnd_idx] = deepcopy(
                self._alleletype['trans_allele'][rnd_idx])

        # cross over aggregate operation
        if random.random() < 0.5:
            para.alleletype['weight_allele'] = deepcopy(
                self._alleletype['weight_allele'])
        else:
            para.alleletype['weight_allele'] = deepcopy(
                other._alleletype['weight_allele'])

        # cross over binary operation
        if random.random() < 0.5:
            para.alleletype['dists_allele'] = deepcopy(
                self._alleletype['dists_allele'])
        else:
            para.alleletype['dists_allele'] = deepcopy(
                other._alleletype['dists_allele'])
        return para
