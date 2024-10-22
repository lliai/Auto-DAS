import argparse
import gc

import numpy as np
import torch

from diswotv2.api.nas101_api import (NB101API, get_nb101_model_and_acc,
                                     get_nb101_teacher)
from diswotv2.datasets.cifar10 import get_cifar10_dataloaders
from diswotv2.searchspace.instincts import LinearInstinct
from diswotv2.searchspace.interactions import ParaInteraction
from diswotv2.utils.misc import all_same, seed_all


def compute_fitness_score(img, label, tmodel, smodel):
    """ compute fitness score of a model"""
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


def random_search(img, label, api, iterations):
    """ random search for the best model"""
    best_hash = None
    best_acc = 0
    best_model = None
    best_score = 0

    teacher = get_nb101_teacher()
    if torch.cuda.is_available():
        teacher = teacher.cuda()

    for i in range(iterations):
        _hash = api.random_hash()
        model, acc = get_nb101_model_and_acc(_hash)
        if torch.cuda.is_available():
            model = model.cuda()

        score = compute_fitness_score(img, label, teacher, model)

        if score > best_score:
            best_hash = _hash
            best_acc = acc
            best_model = model
            best_score = score

        print(
            f' * iteration {i}: hash: {best_hash}, gt acc: {acc}, best: {score}'
        )

        del model
        gc.collect()
        torch.cuda.empty_cache()

    return best_hash, best_acc, best_model, best_score


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
    print('dataloader built')

    # get img, label
    img, label = next(iter(train_loader))
    if torch.cuda.is_available():
        img = img.cuda()
        label = label.cuda()

    # build api
    api = NB101API(path=args.benchmark, verbose=False)

    # random search
    best_hash, best_acc, best_model, best_score = random_search(
        img, label, api, args.iterations)
    print(
        f'best hash: {best_hash}, best acc: {best_acc}, best score: {best_score}'
    )
