import argparse
import csv
import gc
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from diswotv2.api.api import DisWOTAPI
from diswotv2.datasets.cifar100 import get_cifar100_dataloaders
from diswotv2.models import resnet110
from diswotv2.models.candidates.mutable import mutable_resnet20
from diswotv2.searchspace.instincts import LinearInstinct
from diswotv2.searchspace.interactions import ParaInteraction
from diswotv2.utils.misc import all_same, seed_all
from diswotv2.utils.rank_consistency import kendalltau, pearson, spearman

api = DisWOTAPI(
    './data/diswotv2_dict_4e1ea4a046d7e970b9151d8e23332ec5.pkl', verbose=True)


def compute_rank_consistency(img,
                             label,
                             instinct,
                             interaction,
                             scale: bool = False):
    """Compute rank consistency of the interaction.

    Args:
        img (torch.Tensor): The image tensor.
        label (torch.Tensor): The label tensor.
        interaction (Interaction): The interaction.
    """

    # check memory whether have the rank info.
    if instinct.reward_score is not None:
        return instinct.reward_score
    if interaction.reward_score is not None:
        return interaction.reward_score

    gt_list = []
    zc1_list = []  # instinct
    zc2_list = []  # interaction

    tm = resnet110(num_classes=100)

    # to cuda
    if torch.cuda.is_available():
        tm = tm.cuda()

    # traverse the search space in api
    for struct, acc in iter(api):
        # get the model
        sm = mutable_resnet20(struct, num_classes=100)
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


def rnd_search_jointly(img, label, iterations=1000, scale=False):
    """Random search jointly for `iterations` times."""

    # record the best jointly
    best_rk = -1
    best_instinct = None
    best_interaction = None
    best_sp = -1
    best_ps = -1

    # record the procedure of rank
    csv_save_data = []

    for i in range(iterations):
        interaction = ParaInteraction()
        instinct = LinearInstinct()
        kd, sp, ps = compute_rank_consistency(img, label, instinct,
                                              interaction, scale)
        if best_rk < kd:
            best_rk = kd
            best_instinct = instinct
            best_interaction = interaction

        print(
            f'iteration: {i}, best_rk: {best_rk:.4f}, kd: {kd:.4f} sp: {sp:.4f} ps: {ps:.4f}'
        )

        # record
        csv_save_data.append({
            'iteration': i,
            'kendall_tau': best_rk,
            'spearman_rho': best_sp,
            'pearson_rho': best_ps,
        })

    # post process
    idxs, rk_kd, rk_sp, rk_ps = [], [], [], []
    for _dict in csv_save_data:
        idxs.append(_dict['iteration'])
        rk_kd.append(_dict['kendall_tau'])
        rk_sp.append(_dict['spearman_rho'])
        rk_ps.append(_dict['pearson_rho'])
    save_name = f'random_jointly_{type(best_instinct).__name__}_{iterations}_{random.randint(0, 1000)}'

    post_process(idxs, rk_kd, save_name, 'kendall_tau')
    post_process(idxs, rk_sp, save_name, 'spearman_rho')
    post_process(idxs, rk_ps, save_name, 'pearson_rho')

    return best_interaction, best_instinct, best_rk


def post_process(idxs, rks, save_name, rank_name):
    plt.grid()
    # rank correlation
    plt.plot(idxs, rks)
    plt.xlabel('Iteration')
    plt.ylabel('kendall tau')
    plt.savefig(f'./output/rnd_search_jointly/{save_name}_{rank_name}.png')

    # save idx and rks into csv file
    with open(f'./output/rnd_search_jointly/{save_name}_{rank_name}.csv',
              'w') as f:
        writer = csv.writer(f)
        writer.writerows(zip(idxs, rks))


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
        img, label, iterations=args.iterations, scale=args.scale)

    print('random search jointly end...')
    print(f' * best_rk: {best_rk:.4f}')
    print(f' * best_interaction: {best_interaction}')
    print(f' * best_instinct: {best_instinct}')
