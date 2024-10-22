import argparse
import gc

import numpy as np
import torch
import torch.nn.functional as F

from diswotv2.api.nas201_api import (NB201KDAPI, get_network_by_index,
                                     get_teacher_best_model,
                                     random_sample_and_get_gt)
from diswotv2.datasets.cifar100 import get_cifar100_dataloaders
from diswotv2.predictor.pruners import predictive
from diswotv2.utils.misc import all_same, seed_all
from diswotv2.utils.rank_consistency import kendalltau, pearson, spearman


def compute_rank_consistency(dataloader,
                             scale: bool = False,
                             sample_num: int = 50,
                             api=None,
                             zc_name: str = 'zen'):
    """Compute rank consistency """

    gt_list = []
    zc_list = []  # instinct

    dataload_info = ['random', 3, 10]  # last one is num_classes

    # traverse the search space in api
    for i in range(sample_num):
        # get the model
        if api is None:
            sm, acc = random_sample_and_get_gt()
        else:
            rnd_idx = api.random_idx()
            acc = api.get_kd_acc(rnd_idx)  # kd acc
            sm = get_network_by_index(rnd_idx)

        # to cuda
        if torch.cuda.is_available():
            sm = sm.cuda()

        # get the score
        score = predictive.find_measures(
            sm,
            dataloader,
            dataload_info,
            measure_names=[zc_name],
            loss_fn=F.cross_entropy,
            device=torch.device('cpu'))

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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # scale two score list to [0, 1] and then add them .
    if scale:
        zc_list = (zc_list - np.min(zc_list)) / \
            (np.max(zc_list) - np.min(zc_list))

    # compute rank consistency
    kd = kendalltau(gt_list, zc_list)
    sp = spearman(gt_list, zc_list)
    ps = pearson(gt_list, zc_list)

    return kd, sp, ps


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
    parser.add_argument('--zc_name', type=str, default='zen')

    # sample_num for rank consistency
    parser.add_argument('--sample_num', type=int, default=50)

    # benchmark file for nb201 kd
    parser.add_argument(
        '--benchmark',
        type=str,
        default='./data/nb201_kd_dict_1dd544f95b3094a251a0815d3a616dff.pkl',
        help='the benchmark file for nb201 kd')

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

    api = NB201KDAPI(args.benchmark)

    kd, sp, ps = compute_rank_consistency(
        train_loader, scale=args.scale, sample_num=50, zc_name=args.zc_name)
    print(f'kendalltau: {kd} spearman: {sp} pearson: {ps}')
