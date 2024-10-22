import argparse
import gc
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from diswotv2.api.nas201_api import (get_teacher_best_model,
                                     random_sample_and_get_gt)
from diswotv2.datasets.cifar100 import get_cifar100_dataloaders
from diswotv2.searchspace.instincts import LinearInstinct
from diswotv2.searchspace.interactions import ParaInteraction
from diswotv2.utils.misc import all_same, seed_all
from diswotv2.utils.rank_consistency import kendalltau, pearson, spearman


def compute_rank_consistency(img: Tensor,
                             label: Tensor,
                             instinct: LinearInstinct,
                             interaction: ParaInteraction,
                             scale: bool = False,
                             sample_num: int = 50):
    """Compute rank consistency of the search space.

    Args:
        img (Tensor): input image
        label (Tensor): ground truth label
        instinct (LinearInstinct): instinct
        interaction (ParaInteraction): interaction
        scale (bool, optional): scale the scores. Defaults to False.
        sample_num (int, optional): number of samples. Defaults to 50.

    Returns:
        float: kendalltau
        float: spearman
        float: pearson
    """
    # check memory whether have the rank info.
    if instinct.reward_score is not None:
        return instinct.reward_score
    if interaction.reward_score is not None:
        return interaction.reward_score

    gt_list = []
    zc1_list = []  # instinct
    zc2_list = []  # interaction

    tm = get_teacher_best_model()

    # to cuda
    if torch.cuda.is_available():
        tm = tm.cuda()

    # traverse the search space in api
    for i in range(sample_num):
        # get the model
        sm, acc = random_sample_and_get_gt()
        # to cuda
        if torch.cuda.is_available():
            sm = sm.cuda()

        # compute score of the jointly
        score1 = interaction(img, label, tm, sm)
        score2 = instinct(img, label, sm)

        # exception handling
        if score1 == -1 or score2 == -1:
            return -1, -1, -1

        if np.isnan(score1) or np.isnan(score2):
            return -1, -1, -1

        # early stop
        if len(zc1_list) > 3 and all_same(zc1_list):
            return -1, -1, -1
        if len(zc2_list) > 3 and all_same(zc2_list):
            return -1, -1, -1

        # record
        gt_list.append(acc)
        zc1_list.append(score1)
        zc2_list.append(score2)

        # gabage collection
        del sm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # gabage collection
    del tm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # scale two score list to [0, 1] and then add them .
    if scale:
        zc1_list = (zc1_list - np.min(zc1_list)) / \
            (np.max(zc1_list) - np.min(zc1_list))
        zc2_list = (zc2_list - np.min(zc2_list)) / \
            (np.max(zc2_list) - np.min(zc2_list))
    zc_list = [x + y for x, y in zip(zc1_list, zc2_list)]

    # compute rank consistency
    kd = kendalltau(gt_list, zc_list)
    sp = spearman(gt_list, zc_list)
    ps = pearson(gt_list, zc_list)

    if np.isnan(sp):
        return -1, -1, -1
    else:
        instinct.reward_score = [kd, sp, ps]
        interaction.reward_score = [kd, sp, ps]

    return kd, sp, ps


def rnd_search_jointly(img: Tensor,
                       label: Tensor,
                       iterations: int = 1000,
                       logger=None,
                       scale: bool = False,
                       sample_num: int = 50):
    """Random search jointly for `iterations` times.

    Args:
        img (Tensor): The image tensor.
        label (Tensor): The label tensor.
        iterations (int, optional): The number of iterations. Defaults to 1000.
        logger ([type], optional): The logger. Defaults to None.
        scale (bool, optional): Whether to scale the score to [0, 1]. Defaults to False.
        sample_num (int, optional): The number of samples. Defaults to 50.

    """

    # record the best jointly
    best_rk = -1
    best_instinct = None
    best_interaction = None

    # record the procedure of rank
    iter_list = []
    rank_list = []

    for i in range(iterations):
        interaction = ParaInteraction()
        instinct = LinearInstinct()
        kd, sp, ps = compute_rank_consistency(img, label, instinct,
                                              interaction, scale, sample_num)
        if best_rk < kd:
            best_rk = kd
            best_instinct = instinct
            best_interaction = interaction

        if logger is None:
            print(
                f'iteration: {i}, best_rk: {best_rk:.4f}, kd: {kd:.4f} sp: {sp:.4f} ps: {ps:.4f}'
            )
        else:
            print(
                f'iteration: {i}, best_rk: {best_rk:.4f}, kd: {kd:.4f} sp: {sp:.4f} ps: {ps:.4f}'
            )

        # record
        iter_list.append(i)
        rank_list.append(best_rk)

    # plot the procedure of rank
    plt.grid()
    plt.plot(iter_list, rank_list)
    plt.xlabel('iteration')
    plt.ylabel('Kendall Tau')
    plt.savefig(
        f'./output/rnd_search_jointly_{iterations}_{random.randint(0, 100)}.png'
    )

    return best_interaction, best_instinct, best_rk


if __name__ == '__main__':
    # add argparse to support iteration
    parser = argparse.ArgumentParser('random search jointly')
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save', type=str, default='./output')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--scale', type=bool, default=False)

    # sample_num for rank consistency
    parser.add_argument('--sample_num', type=int, default=50)

    args = parser.parse_args()

    # seed all
    seed_all(args.seed)

    # build dataloader
    train_loader, val_loader = get_cifar100_dataloaders(
        args.data_path, args.batch_size)

    # get img, label
    img, label = next(iter(train_loader))

    # to cuda
    if torch.cuda.is_available():
        img = img.cuda()
        label = label.cuda()

    # random search
    print('random search jointly begin...')
    best_interaction, best_instinct, best_rk = rnd_search_jointly(
        img,
        label,
        iterations=args.iterations,
        scale=args.scale,
        sample_num=args.sample_num)

    print('random search jointly end...')
    print(f' * best_rk: {best_rk:.4f}')
    print(f' * best_interaction: {best_interaction}')
    print(f' * best_instinct: {best_instinct}')
