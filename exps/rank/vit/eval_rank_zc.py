import argparse
import gc

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from diswotv2.api.vit_api import DisWOT_API_VIT
from diswotv2.datasets.chaoyang import Chaoyang
from diswotv2.datasets.cifar100 import vit_cifar100_dataloaders
from diswotv2.datasets.flowers import Flowers
from diswotv2.models.candidates.mutable.vit import AutoFormer
from diswotv2.predictor.pruners import predictive
from diswotv2.utils.misc import seed_all
from diswotv2.utils.rank_consistency import kendalltau, pearson, spearman


def compute_rank_consistency(dataloader,
                             scale: bool = False,
                             api=None,
                             sample_num: int = 50,
                             NUM_CLASSES: int = 10,
                             zc_name: str = 'zen'):
    """Compute rank consistency via vanilla zc."""

    dataload_info = ['random', 3, NUM_CLASSES]

    # asserts
    assert api is not None, 'nb101_dict is None'

    gt_list = []
    zc_list = []  # instinct

    # traverse the search space in api
    for i, (struct, acc) in enumerate(iter(api)):
        # get the model
        sm = AutoFormer(struct, num_classes=100)
        # to cuda
        if torch.cuda.is_available():
            sm = sm.cuda()

        # get the score
        if isinstance(zc_name, str):
            zc_name = [zc_name]
        score = predictive.find_measures(
            sm,
            dataloader,
            dataload_info,
            measure_names=zc_name,
            loss_fn=F.cross_entropy,
            device=torch.device('cpu'))

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
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save', type=str, default='./output')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--scale', type=bool, default=False)
    parser.add_argument('--sample_num', type=int, default=20)
    parser.add_argument('--zc_name', type=str, default='zen')

    parser.add_argument(
        '--api_path', type=str, default='./data/diswotv2_autoformer.pth')
    parser.add_argument(
        '--dataset',
        type=str,
        default='c100',
        choices='c100, flower, chaoyang')

    args = parser.parse_args()

    # seed all
    seed_all(args.seed)

    # build dataloader
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
    print('dataloader built')

    # get img, label
    img, label = next(iter(train_loader))
    if torch.cuda.is_available():
        img = img.cuda()
        label = label.cuda()

    # build api
    api = DisWOT_API_VIT(args.api_path, mode='kd', dataset=args.dataset)

    kd, sp, ps = compute_rank_consistency(train_loader, args.scale, api,
                                          args.sample_num, args.zc_name)

    print(f'kendalltau: {kd} spearman: {sp} pearson: {ps}')
