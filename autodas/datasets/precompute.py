import argparse
import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from torchvision.transforms import Compose
from tqdm import tqdm

from diswotv2.datasets.transforms import create_test_transform
from diswotv2.models import model_dict


def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return f'{segments[0]}_{segments[1]}_{segments[2]}'


def load_teacher(model_path, n_cls=100):
    model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    model.load_state_dict(
        torch.load(model_path, map_location=torch.device('cpu'))['model'])
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='resnet110')
    parser.add_argument(
        '--ckpt',
        type=str,
        default='./save/models/resnet110_vanilla/ckpt_epoch_240.pth')
    args = parser.parse_args()

    save_dir = './data'
    os.makedirs(save_dir, exist_ok=True)

    transforms = create_test_transform(
        mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2565, 0.2761),
        img_size=32)

    transform = Compose(transforms)
    dataset = CIFAR100(
        root='data', train=True, transform=transform, download=True)
    loader = DataLoader(dataset, batch_size=32)

    model = load_teacher(args.ckpt, n_cls=100)
    model.load_state_dict(torch.load(args.ckpt, map_location='cpu')['model'])
    if torch.cuda.is_available():
        model.cuda()

    # NOTE: make sure that the teacher model have features.

    feat_dict = defaultdict(list)
    for img, _ in tqdm(loader):
        if torch.cuda.is_available():
            img = img.cuda()
        with torch.no_grad():
            model(img)

        for i, feat in enumerate(model.features):
            # N, _, H, W = feat.shape
            feat = feat.cpu()
            feat_dict[f'layer_{i}'].append(feat)

    for k in feat_dict:
        feat_dict[k] = torch.cat(feat_dict[k], dim=0).numpy()

    np.savez(
        os.path.join(save_dir, 'cifar100_resnet110_pre_features.npz'),
        **feat_dict)


if __name__ == '__main__':
    main()
