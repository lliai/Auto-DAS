import argparse
import gc

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

    interaction = ParaInteraction()
    interaction.update_alleletype(
        "ALLELE# in:['k2']~       trans:['trans_relu', 'trans_sigmoid', 'trans_abs']~     weig:['w100_teacher_student']~  dist:['kl_T8']"
    )

    instinct = LinearInstinct()
    instinct.update_genotype(
        'INPUT:(grad)UNARY:|logsoftmax|no_op|frobenius_norm|normalized_sum|')

    kd, sp, ps = compute_rank_consistency(img, label, instinct, interaction,
                                          args.scale)

    print(f'kendalltau: {kd} spearman: {sp} pearson: {ps}')
