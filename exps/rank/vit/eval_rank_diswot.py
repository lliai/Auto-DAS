import argparse
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from diswotv2.api.vit_api import DisWOT_API_VIT
from diswotv2.datasets.chaoyang import Chaoyang
from diswotv2.datasets.cifar100 import vit_cifar100_dataloaders
from diswotv2.datasets.flowers import Flowers
from diswotv2.losses import (Attention, Correlation, HintLoss, ICKDLoss,
                             NSTLoss, RKDLoss, Similarity)
# from diswotv2.datasets.cifar10 import get_cifar10_dataloaders
# from diswotv2.datasets.cifar100 import get_cifar100_dataloaders
# from diswotv2.datasets.imagenet16 import get_imagenet16_dataloaders
from diswotv2.models import resnet56
from diswotv2.models.candidates.mutable.vit import PIT, AutoFormer
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


def compute_diswot_procedure(img,
                             api=None,
                             sample_num: int = 50,
                             num_classes=100,
                             api_path='autoformer'):

    # grad-cam not cam
    tnet = resnet56(num_classes=num_classes)

    criterion_ickd = ICKDLoss()

    gt_list = []
    zcs_list = []

    for i, (struct, acc) in enumerate(iter(api)):
        print(i)
        if i > sample_num:
            break  # early stop

        # get the model
        if 'autoformer' in api_path:
            snet = AutoFormer(struct, num_classes=num_classes)
        else:
            snet = PIT(struct, num_classes=num_classes)

        gt_list.append(float(acc))

        sfeature, slogits = snet(img, is_feat=True)

        t_img = F.interpolate(
            img, size=(32, 32), mode='bilinear', align_corners=False)
        tfeature, tlogits = tnet(t_img, is_feat=True)

        def inter_fd(f_s, f_t):
            s_C, t_C, s_H, t_H = f_s.shape[1], f_t.shape[1], f_s.shape[
                2], f_t.shape[2]
            if s_H > t_H:
                f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
            elif s_H < t_H:
                f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
                f_s = F.interpolate(
                    f_s, size=(t_H, t_H), mode='bilinear', align_corners=True)

            return f_s[:, 0:min(s_C, t_C), :, :], f_t[:,
                                                      0:min(s_C, t_C
                                                            ), :, :].detach()

        teacher_feat = tfeature[-2]
        student_feat = sfeature[-2]
        teacher_feat, student_feat = inter_fd(teacher_feat, student_feat)

        # import pdb; pdb.set_trace()
        # score_sp = -1 * criterion_sp(teacher_feat,
        #                              student_feat)[0].detach().numpy()

        score_ickd = -1 * criterion_ickd([teacher_feat],
                                         [student_feat])[0].detach().numpy()
        result = score_ickd
        zcs_list.append(result if isinstance(result, float) else result[0])

        # gabage collection
        del snet
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # def min_max_scale(x):
    #     return (x - np.min(x)) / (np.max(x) - np.min(x))

    # zcs_list = min_max_scale(zcs_list)

    print(
        f'kd: {kendalltau(zcs_list,gt_list):.4f} sp: {spearman(zcs_list, gt_list):.4f} ps: {pearson(zcs_list, gt_list):.4f}'
    )
    kd = kendalltau(gt_list, zcs_list)
    sp = spearman(gt_list, zcs_list)
    ps = pearson(gt_list, zcs_list)
    return kd, sp, ps


def compute_kd_procedure(img, api=None, sample_num=50, kd_loss='nst'):

    # grad-cam not cam
    tnet = resnet56(num_classes=100)

    gt_list = []
    zcs_list = []

    for i, (struct, acc) in enumerate(iter(api)):
        print(i)
        if i > sample_num:
            break  # early stop

        # get the model
        snet = AutoFormer(struct, num_classes=100)

        gt_list.append(float(acc))

        sfeature, slogits = snet(img, is_feat=True)

        t_img = F.interpolate(
            img, size=(32, 32), mode='bilinear', align_corners=False)
        tfeature, tlogits = tnet(t_img, is_feat=True)

        def attention_transform(feat):
            return F.normalize(feat.pow(2).mean(1).view(feat.size(0), -1))

        teacher_feat = tfeature[-2]
        student_feat = sfeature[-2]
        teacher_feat = attention_transform(teacher_feat)
        student_feat = attention_transform(student_feat)

        # kd,nst,cc,ickd,rkd,at,fitnet,sp
        if kd_loss == 'kd':
            criterion = nn.KLDivLoss()
            result = criterion(teacher_feat, student_feat)
        elif kd_loss == 'nst':
            criterion = NSTLoss()
            result = criterion(teacher_feat, student_feat)
        elif kd_loss == 'cc':
            criterion = Correlation()
            result = criterion(teacher_feat, student_feat)
        elif kd_loss == 'ickd':
            criterion = ICKDLoss()
            result = criterion([teacher_feat], [student_feat])
        elif kd_loss == 'rkd':
            criterion = RKDLoss()
            result = criterion([teacher_feat], [student_feat])
        elif kd_loss == 'at':
            criterion = Attention()
            result = criterion([teacher_feat], [student_feat])
        elif kd_loss == 'fitnet':
            criterion = HintLoss()
            result = criterion([teacher_feat], [student_feat])
        elif kd_loss == 'sp':
            criterion = Similarity()
            result = criterion(teacher_feat, student_feat)
        else:
            raise NotImplementedError

        result = -1 * result.detach().numpy()
        zcs_list.append(result if isinstance(result, float) else result[0])

        # gabage collection
        del snet
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        zcs_list.append(result if isinstance(result, float) else result[0])

        # gabage collection
        del snet
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(
        f'kd: {kendalltau(zcs_list,gt_list):.4f} sp: {spearman(zcs_list, gt_list):.4f} ps: {pearson(zcs_list, gt_list):.4f}'
    )
    kd = kendalltau(gt_list, zcs_list)
    sp = spearman(gt_list, zcs_list)
    ps = pearson(gt_list, zcs_list)
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
    parser.add_argument('--data_path', type=str, default='./data/chaoyang')
    parser.add_argument('--scale', type=bool, default=False)
    parser.add_argument('--sample_num', type=int, default=20)
    parser.add_argument('--zc_name', type=str, default='zen')
    parser.add_argument(
        '--api_path', type=str, default='./data/diswotv2_autoformer.pth')
    parser.add_argument(
        '--dataset',
        type=str,
        default='chaoyang',
        choices='c100, flowers, chaoyang')
    parser.add_argument('--kd_loss', type=str, default='nst')

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
    if args.dataset in ['flowers', 'chaoyang']:
        img, label, _ = next(iter(train_loader))
    elif args.dataset == 'c100':
        img, label = next(iter(train_loader))

    # build api
    api = DisWOT_API_VIT(args.api_path, mode='kd', dataset=args.dataset)

    kd, sp, ps = compute_diswot_procedure(img, api, args.sample_num,
                                          num_classes, args.api_path)

    print(f'kendalltau: {kd} spearman: {sp} pearson: {ps}')
