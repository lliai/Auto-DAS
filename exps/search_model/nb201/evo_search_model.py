import argparse
import datetime
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from nas_201_api import NASBench201API

from diswotv2.api.nas201_api import (get_network_by_archstr,
                                     get_teacher_best_model,
                                     query_gt_by_arch_str)
from diswotv2.datasets.cifar10 import get_cifar10_dataloaders
from diswotv2.helper.utils.flop_benchmark import get_model_infos
from diswotv2.searchspace.instincts import LinearInstinct
from diswotv2.searchspace.interactions import ParaInteraction
from diswotv2.utils.misc import seed_all

nb201_api = NASBench201API(
    file_path_or_dict='./data/NAS-Bench-201-v1_0-e61699.pth', verbose=False)


def compute_fitness_score(img, label, tmodel, smodel):
    """compute the fitness score of the model.

    Args:
        img (torch.Tensor): [B, C, H, W]
        label (torch.Tensor): [B]
        tmodel (torch.nn.Module): teacher model
        smodel (torch.nn.Module): student model
    """

    instinct = LinearInstinct()
    instinct.update_genotype(
        'INPUT:(virtual_grad)UNARY:|abslog|sigmoid|normalized_sum|invert|')

    interaction = ParaInteraction()
    interaction.update_alleletype(
        "ALLELE# in:['k3']~       trans:['trans_drop', 'trans_multi_scale_r4', 'trans_pow2']~     weig:['w100_teacher_student']~  dist:['l1_loss']"
    )

    score1 = interaction(img, label, tmodel, smodel)
    score2 = instinct(img, label, smodel)

    return score1 + score2


def to_arch_str(arch_list: list):
    """convert arch string. """
    assert isinstance(arch_list, list), 'invalid arch_list type : {:}'.format(
        type(arch_list))

    strings = []
    for node_info in arch_list:
        string = '|'.join([x[0] + '~{:}'.format(x[1]) for x in node_info])
        string = f'|{string}|'
        strings.append(string)
    return '+'.join(strings)


def mutate_nb201(arch_str: str) -> str:
    """mutate the arch in nas-bench-201
    arch_list = [(('avg_pool_3x3', 0),), (('skip_connect', 0), ('none', 1)), (('none', 0), ('none', 1), ('skip_connect', 2))]

    - random sample a position from six positions, eg: avg_pool_3x3
    - replace it with a randomly sampled operation, eg: none
    - candidate operation is:
        - nor_conv_1x1
        - nor_conv_3x3
        - avg_pool_3x3
        - none
        - skip_connect

    return [(('none', 0),), (('skip_connect', 0), ('none', 1)), (('none', 0), ('none', 1), ('skip_connect', 2))]
    """
    if isinstance(arch_str, str):
        arch_list = nb201_api.str2lists(arch_str)
    else:
        raise TypeError('invalid arch_str type : {:}'.format(type(arch_str)))

    # convert items in arch_list to list
    tmp_list = []
    for layer in arch_list:
        tmp_ = []
        for oper in layer:
            tmp_.append(list(oper))
        tmp_list.append(tmp_)

    # candidate position
    operations = [
        'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3', 'skip_connect', 'none'
    ]

    # sample layer from [0, 1, 2]
    layer_idx = random.randint(0, 2)

    # sample operation from operations
    op_idx = random.randint(0, len(tmp_list[layer_idx]) - 1)

    try:
        tmp_list[layer_idx][op_idx][0] = operations[random.randint(
            0,
            len(operations) - 1)]
    except IndexError:
        import pdb
        pdb.set_trace()

    return to_arch_str(tmp_list)


def crossover_nb201(arch_str1: str, arch_str2: str) -> str:
    """ make cross over between two archs"""
    if isinstance(arch_str1, str):
        arch_list1 = nb201_api.str2lists(arch_str1)
    else:
        raise TypeError('invalid arch_str1 type : {:}'.format(type(arch_str1)))

    if isinstance(arch_str2, str):
        arch_list2 = nb201_api.str2lists(arch_str2)
    else:
        raise TypeError('invalid arch_str2 type : {:}'.format(type(arch_str2)))

    # convert items in arch_list to list
    tmp_list1 = []
    tmp_list2 = []
    for layer in arch_list1:
        tmp_ = []
        for oper in layer:
            tmp_.append(list(oper))
        tmp_list1.append(tmp_)

    for layer in arch_list2:
        tmp_ = []
        for oper in layer:
            tmp_.append(list(oper))
        tmp_list2.append(tmp_)

    # sample layer from [0, 1, 2]
    layer_idx = random.randint(0, 2)

    # sample operation from operations
    op_idx = random.randint(0, len(tmp_list1[layer_idx]))

    try:
        tmp_list1[layer_idx][op_idx][0] = tmp_list2[layer_idx][op_idx][0]
    except IndexError:
        import pdb
        pdb.set_trace()

    return to_arch_str(tmp_list1)


class NB201Object:

    def __init__(self, arch_str: str):
        super().__init__()
        self.arch_str = arch_str
        self._score = -1

    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, value):
        self._score = value


def evolution_search(img,
                     label,
                     model_size=5,
                     iterations=1000,
                     popu_size=100,
                     elite_size=10,
                     mutation_rate=0.5,
                     crossover_rate=0.5,
                     seed=42,
                     flops=1000):
    """ evolution search for nas-bench-201

    Args:
        img (torch.Tensor): input image
        label (torch.Tensor): input label
        model_size (int): model size
        iterations (int, optional): number of iterations. Defaults to 1000.
        popu_size (int, optional): population size. Defaults to 100.
        elite_size (int, optional): elite size. Defaults to 10.
        mutation_rate (float, optional): mutation rate. Defaults to 0.5.
        crossover_rate (float, optional): crossover rate. Defaults to 0.5.
        seed (int, optional): seed. Defaults to 42.

    """

    populations = []

    tmodel = get_teacher_best_model()

    if torch.cuda.is_available():
        tmodel = tmodel.cuda()

    # initilize population
    for i in range(popu_size):
        # build arch_obj
        choiced_index = random.choice(list(range(15625)))
        arch_str = nb201_api.arch(choiced_index)
        arch_obj = NB201Object(arch_str)
        smodel = get_network_by_archstr(arch_str)

        FLOPs, Param = get_model_infos(tmodel, shape=(1, 3, 32, 32))
        if FLOPs > flops:
            choiced_index = random.choice(list(range(15625)))
            arch_str = nb201_api.arch(choiced_index)
            arch_obj = NB201Object(arch_str)
            smodel = get_network_by_archstr(arch_str)
            FLOPs, Param = get_model_infos(tmodel, shape=(1, 3, 32, 32))

            if FLOPs > flops:
                continue

        if torch.cuda.is_available():
            smodel = smodel.cuda()

        # compute score
        stu_score = compute_fitness_score(img, label, tmodel, smodel)
        arch_obj.score = stu_score
        populations.append(arch_obj)

        print(f'initilize {i}th population, score: {stu_score}')

        del smodel  # release memory

    # record for iters and best score
    iters_list, best_score_list = [], []
    best_score = -1  # initilize best score
    best_obj = None  # initilize best obj

    # start evolution
    for i in range(iterations):
        iters_list.append(i)
        scores = [obj.score for obj in populations]

        # select the best one
        scores = np.array(scores)
        argidxs = np.argsort(scores)[::-1]

        best_score_list.append(scores[argidxs[0]])
        if best_score < scores[argidxs[0]]:
            best_score = scores[argidxs[0]]
            best_obj = populations[argidxs[0]]
            assert best_obj.score == best_score

        print(
            f'iter: {i}, best score: {best_score}, gt score: {query_gt_by_arch_str(best_obj.arch_str)}'
        )

        best_obj = populations[argidxs[0]]

        # remove the worse one
        del populations[argidxs[-1]]

        # mutate the best one
        mutated_archstr = mutate_nb201(best_obj.arch_str)
        mutated_obj = NB201Object(mutated_archstr)
        mutated_model = get_network_by_archstr(mutated_archstr)
        if torch.cuda.is_available():
            mutated_model = mutated_model.cuda()
        mutated_score = compute_fitness_score(img, label, tmodel,
                                              mutated_model)
        mutated_obj.score = mutated_score

        # random sample one
        random_archstr = nb201_api.arch(random.choice(list(range(15625))))
        random_obj = NB201Object(random_archstr)
        random_model = get_network_by_archstr(random_archstr)
        if torch.cuda.is_available():
            random_model = random_model.cuda()
        random_score = compute_fitness_score(img, label, tmodel, random_model)
        random_obj.score = random_score

        # add better one to the population
        if mutated_score < random_score:
            mutated_obj.score = mutated_score
            populations.append(mutated_obj)
        else:
            random_obj.score = random_score
            populations.append(random_obj)

        del random_model
        del mutated_model

    # gather the scores
    scores = [obj.score for obj in populations]
    argidxs = np.argsort(scores)[::-1]

    print(
        f' * best score: {scores[argidxs[0]]} best arch: {populations[argidxs[0]].arch_str}'
        f' gt score: {query_gt_by_arch_str(populations[argidxs[0]].arch_str)}')

    plt.plot(iters_list, best_score_list)

    current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    plt.savefig(
        f'./output/evolution_search_nb201_procedure_{current_time}.png')

    return best_obj.arch_str, best_obj.score


if __name__ == '__main__':
    # add argparse to support iteration
    parser = argparse.ArgumentParser('random search jointly')
    parser.add_argument('--iterations', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save', type=str, default='./output')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--scale', type=bool, default=False)
    parser.add_argument('--sample_num', type=int, default=50)
    parser.add_argument(
        '--flops', type=float, default=50, help='flops limit: 50 or 100M')

    # benchmark file
    parser.add_argument(
        '--benchmark',
        type=str,
        default='./data/nb101_dict_358bc32bd6537af8b13ed28d260e0c74.pkl')

    args = parser.parse_args()

    # seed all
    seed_all(args.seed)

    # build dataloader
    train_loader, val_loader = get_cifar10_dataloaders(args.data_path,
                                                       args.batch_size)

    # get img, label
    img, label = next(iter(train_loader))
    if torch.cuda.is_available():
        img = img.cuda()
        label = label.cuda()

    # begin evolution procedure

    arch_str, score = evolution_search(
        img,
        label,
        model_size=5,
        iterations=args.iterations,
        popu_size=5,
        elite_size=2,
        mutation_rate=0.5,
        crossover_rate=0.5,
        seed=args.seed,
        flops=args.flops)

    print('best arch: {:}'.format(arch_str))
    print('best score: {:}'.format(score))
    print('gt score: {:}'.format(query_gt_by_arch_str(arch_str)))
