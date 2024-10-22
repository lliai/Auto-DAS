import argparse
import gc
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from diswotv2.api.api import DisWOTAPI
from diswotv2.datasets.cifar100 import get_cifar100_dataloaders
from diswotv2.models import resnet110
from diswotv2.models.candidates.mutable import mutable_resnet20
from diswotv2.searchspace.interactions import ParaInteraction
from diswotv2.utils.misc import all_same, seed_all
from diswotv2.utils.rank_consistency import kendalltau, pearson, spearman

api = DisWOTAPI(
    './data/diswotv2_dict_4e1ea4a046d7e970b9151d8e23332ec5.pkl', verbose=True)


def compute_rank_consistency(img, label, interaction):
    """Compute rank consistency of the interaction.

    Args:
        img (torch.Tensor): The image tensor.
        label (torch.Tensor): The label tensor.
        interaction (Interaction): The interaction.

    """
    # check memory whether have the rank info.
    if interaction.reward_score is not None:
        return interaction.reward_score

    gt_list = []
    zc_list = []
    tm = resnet110(num_classes=100)

    # to cuda
    if torch.cuda.is_available():
        tm = tm.cuda()

    # traverse the search space in api
    for struct, acc in iter(api):
        # get the model
        sm = mutable_resnet20(struct)
        # to cuda
        if torch.cuda.is_available():
            sm = sm.cuda()

        # compute score of the interaction
        score = interaction(img, label, tm, sm)

        # exception handling
        if score == -1:
            return -1, -1, -1

        # early stop
        if len(zc_list) > 3 and all_same(zc_list):
            return -1, -1, -1

        # record
        gt_list.append(acc)
        zc_list.append(score)

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

    # compute rank consistency
    kd = kendalltau(gt_list, zc_list)
    sp = spearman(gt_list, zc_list)
    ps = pearson(gt_list, zc_list)

    if np.isnan(sp):
        return -1, -1, -1
    else:
        interaction.reward_score = [kd, sp, ps]

    return kd, sp, ps


def rnd_search_interaction(img, label, iterations=1000):
    """Random search interaction for `iterations` times."""

    best_rk = -1
    best_interaction = None
    iter_list = []
    rank_list = []

    for i in range(iterations):
        interaction = ParaInteraction()
        kd, sp, ps = compute_rank_consistency(img, label, interaction)
        if best_rk < kd:
            best_rk = kd
            best_interaction = interaction

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
        f'./output/rnd_search_interaction_{iterations}_{random.randint(0, 100)}.png'
    )

    return best_interaction, best_rk


if __name__ == '__main__':
    # add argparse to support iteration
    parser = argparse.ArgumentParser('random search interaction')
    parser.add_argument('--iterations', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save', type=str, default='./output')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--data_path', type=str, default='./data')

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
    print('random search interaction begin...')
    best_interaction, best_rk = rnd_search_interaction(
        img, label, iterations=args.iterations)
    print('random search interaction end...')
    print(f' * best rank: {best_rk:.4f}')
    print(f' * best interaction: {best_interaction}')
