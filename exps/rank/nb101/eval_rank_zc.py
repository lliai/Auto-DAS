import argparse
import gc

import numpy as np
import torch
import torch.nn.functional as F

from diswotv2.api.nas101_api import NB101API, get_nb101_model_and_acc
from diswotv2.datasets.cifar10 import get_cifar10_dataloaders
from diswotv2.predictor.pruners import predictive
from diswotv2.utils.misc import seed_all
from diswotv2.utils.rank_consistency import kendalltau, pearson, spearman


def compute_rank_consistency(dataloader,
                             scale: bool = False,
                             api: NB101API = None,
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
    for i in range(sample_num):
        # get the hash
        _hash = api.random_hash()

        # get the model
        sm, acc = get_nb101_model_and_acc(_hash)
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

    # benchmark file
    parser.add_argument(
        '--benchmark',
        type=str,
        default='./data/nb101_kd_dict_9756ff660472a567ebabe535066c0e1f.pkl')
    # './data/nb101_dict_358bc32bd6537af8b13ed28d260e0c74.pkl')

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

    kd, sp, ps = compute_rank_consistency(train_loader, args.scale, api,
                                          args.sample_num, args.zc_name)

    print(f'kendalltau: {kd} spearman: {sp} pearson: {ps}')
