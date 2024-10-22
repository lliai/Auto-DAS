import argparse
import csv
import gc
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from diswotv2.api.vit_api import DisWOT_API_VIT
from diswotv2.datasets.chaoyang import Chaoyang
from diswotv2.datasets.cifar100 import vit_cifar100_dataloaders
from diswotv2.datasets.flowers import Flowers
from diswotv2.models import resnet56
from diswotv2.models.candidates.mutable.vit import PIT, AutoFormer
from diswotv2.searchspace.instincts import LinearInstinct
from diswotv2.searchspace.interactions import ParaInteraction
from diswotv2.utils.misc import all_same, seed_all
from diswotv2.utils.rank_consistency import kendalltau, pearson, spearman


def compute_rank_consistency(img,
                             label,
                             instinct,
                             interaction,
                             api,
                             scale: bool = False,
                             searchspace: str = 'autoformer'):
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

    tm = resnet56(num_classes=100)

    # to cuda
    if torch.cuda.is_available():
        tm = tm.cuda()

    # traverse the search space in api
    for i, (struct, acc) in enumerate(iter(api)):
        if i > 20:
            break  # early stop

        # get the model
        if searchspace == 'autoformer':
            sm = AutoFormer(struct, num_classes=100)
        elif searchspace == 'pit':
            sm = PIT(struct, num_classes=100)
        # to cuda
        if torch.cuda.is_available():
            sm = sm.cuda()

        # compute score of the jointly
        score1 = interaction(img, label, tm, sm, interpolate=True)
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


def rnd_search_jointly(img,
                       label,
                       api,
                       iterations=1000,
                       scale=False,
                       searchspace='autoformer'):
    """Random search jointly for `iterations` times."""

    # record the best jointly
    best_rk = -1
    best_instinct = None
    best_interaction = None
    best_kd = -1
    best_sp = -1
    best_ps = -1

    # record the procedure of rank
    csv_save_data = []

    for i in range(iterations):
        interaction = ParaInteraction()
        instinct = LinearInstinct()
        kd, sp, ps = compute_rank_consistency(img, label, instinct,
                                              interaction, api, scale,
                                              searchspace)
        if best_rk < sp:
            best_rk = sp
            best_kd = kd
            best_sp = sp
            best_ps = ps
            best_instinct = instinct
            best_interaction = interaction

        print(
            f'iteration: {i}, best_rk(sp): {best_rk:.4f}, kd: {kd:.4f} sp: {sp:.4f} ps: {ps:.4f}'
        )

        # record
        csv_save_data.append({
            'iteration': i,
            'kendall_tau': best_kd,
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
    parser.add_argument('--iterations', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save', type=str, default='./output')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument(
        '--api_path', type=str, default='./data/diswotv2_autoformer.pth')
    parser.add_argument(
        '--dataset',
        type=str,
        default='c100',
        choices='c100, flower, chaoyang')
    parser.add_argument('--scale', type=bool, default=False)
    parser.add_argument(
        '--searchspace', type=str, default='autoformer')  # or pit

    args = parser.parse_args()

    # seed all
    seed_all(args.seed)

    if args.dataset == 'c100':
        # build dataloader
        train_loader, val_loader = vit_cifar100_dataloaders(
            args.data_path, args.batch_size)
    elif args.dataset == 'flower':
        train_dataset = Flowers(args.data_path, split='train')
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4)
    elif args.dataset == 'chaoyang':
        train_dataset = Chaoyang(args.data_path, split='train')
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4)
    else:
        raise NotImplementedError

    # get img, label
    img, label = next(iter(train_loader))

    # to cuda
    if torch.cuda.is_available():
        img = img.cuda()
        label = label.cuda()

    api = DisWOT_API_VIT(args.api_path, mode='kd', dataset=args.dataset)

    # random search
    print('random search jointly begin...')
    best_interaction, best_instinct, best_rk = rnd_search_jointly(
        img,
        label,
        api,
        iterations=args.iterations,
        scale=args.scale,
        searchspace=args.searchspace)

    print('random search jointly end...')
    print(f' * best_rk: {best_rk:.4f}')
    print(f' * best_interaction: {best_interaction}')
    print(f' * best_instinct: {best_instinct}')
