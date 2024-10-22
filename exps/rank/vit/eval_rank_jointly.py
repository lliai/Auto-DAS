import argparse
import gc

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
                             scale: bool = False,
                             api=None,
                             sample_num: int = 50,
                             api_path='autoformer',
                             num_classes=100):
    """Compute rank consistency of two scores.

    Args:
        img (torch.Tensor): [B, C, H, W]
        label (torch.Tensor): [B]
        instinct (LinearInstinct): instinct
        interaction (ParaInteraction): interaction
        scale (bool, optional): scale two score list to [0, 1] and then add
            them. Defaults to False.
        api (NB101API, optional): api. Defaults to None.
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

    # asserts
    assert api is not None, 'nb101_dict is None'

    gt_list = []
    zc1_list = []  # instinct
    zc2_list = []  # interaction

    tm = resnet56(num_classes=num_classes)

    # to cuda
    if torch.cuda.is_available():
        tm = tm.cuda()

    # traverse the search space in api
    for i, (struct, acc) in enumerate(iter(api)):
        print(i)
        if i > 50:
            break  # early stop

        # get the model
        if 'autoformer' in api_path:
            sm = AutoFormer(struct, num_classes=num_classes)
        else:
            sm = PIT(struct, num_classes=num_classes)
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


if __name__ == '__main__':
    # add argparse to support iteration
    parser = argparse.ArgumentParser('random search jointly')
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save', type=str, default='./output')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--scale', type=bool, default=False)
    parser.add_argument('--sample_num', type=int, default=20)

    parser.add_argument(
        '--dataset',
        type=str,
        default='c100',
        choices='c100, flowers, chaoyang')
    parser.add_argument(
        '--api_path', type=str, default='./data/diswotv2_autoformer.pth')
    args = parser.parse_args()

    # seed all
    seed_all(args.seed)

    # build dataloader
    if args.dataset == 'c100':
        # build dataloader
        train_loader, val_loader = vit_cifar100_dataloaders(
            args.data_path, args.batch_size)
        num_classes = 100
    elif args.dataset == 'flowers':
        train_dataset = Flowers(args.data_path, split='train')
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4)
        num_classes = 102
    elif args.dataset == 'chaoyang':
        train_dataset = Chaoyang(args.data_path, split='train')
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4)
        num_classes = 4
    else:
        raise NotImplementedError
    print('dataloader built')

    # get img, label
    if args.dataset == 'c100':
        img, label = next(iter(train_loader))
    else:
        img, label, _ = next(iter(train_loader))
    if torch.cuda.is_available():
        img = img.cuda()
        label = label.cuda()

    # build api
    api = DisWOT_API_VIT(args.api_path, mode='kd', dataset=args.dataset)

    instinct = LinearInstinct()
    instinct.update_genotype(
        'INPUT:(grad)UNARY:|sigmoid|logsoftmax|abslog|no_op|')

    interaction = ParaInteraction()
    interaction.update_alleletype(
        "ALLELE# in:['k3']~       trans:['trans_sigmoid', 'trans_mask', 'trans_softmax_N']~       weig:['w1_teacher']~    dist:['kl_T1']"
    )
    kd, sp, ps = compute_rank_consistency(img, label, instinct, interaction,
                                          args.scale, api, args.sample_num,
                                          args.api_path, num_classes)

    print(f'kendalltau: {kd} spearman: {sp} pearson: {ps}')
