import random

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from timm.data import create_transform
from timm.data.transforms import \
    RandomResizedCropAndInterpolation as _RandomResizedCropAndInterpolation

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class RandomResizedCropAndInterpolation(_RandomResizedCropAndInterpolation):

    def __call__(self, img, features):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation

        out_img = F.resized_crop(img, i, j, h, w, self.size, interpolation)

        i, j, h, w = i / img.size[1], j / \
            img.size[0], h / img.size[1], w / img.size[0]
        out_feats = []
        for feat in features[:-2]:
            feat_h, feat_w = feat.shape[-2:]
            feat = F.resized_crop(
                feat,
                int(i * feat_h),
                int(j * feat_w),
                int(h * feat_h),
                int(w * feat_w),
                size=(feat_h, feat_w))
            out_feats.append(feat)
        out_feats.extend(features[-2:])

        return out_img, out_feats


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):

    def forward(self, img, features):
        if torch.rand(1) < self.p:
            out_img = F.hflip(img)
            out_feats = [F.hflip(feat) for feat in features[:-2]]
            out_feats.extend(features[-2:])
            return out_img, out_feats
        return img, features


def create_train_transform(mean=None,
                           std=None,
                           img_size=None,
                           offline=False,
                           strong_augmentation=False):
    mean = IMAGENET_DEFAULT_MEAN if mean is None else mean
    std = IMAGENET_DEFAULT_STD if std is None else std

    size = (img_size, img_size)
    transform = create_transform(
        input_size=size,
        is_training=True,
        color_jitter=0.4,
        auto_augment='rand-m9-mstd0.5-inc1',
        re_prob=0.25,
        re_mode='pixel',
        re_count=1,
        interpolation='bicubic',
        separate=True,
        mean=mean,
        std=std)
    primary_tfl, secondary_tfl, final_tfl = transform

    if offline:
        primary_tfl = [
            RandomResizedCropAndInterpolation(size, interpolation='bicubic'),
            RandomHorizontalFlip(p=0.5)
        ]

    if not strong_augmentation:
        primary_tfl = transforms.Compose([
            transforms.Resize(size),
            transforms.RandomCrop(size, padding=img_size // 8),
            transforms.RandomHorizontalFlip(p=0.5)
        ])
        secondary_tfl = transforms.Compose([])
        final_tfl = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean), std=torch.tensor(std))
        ])

    return primary_tfl, secondary_tfl, final_tfl


def create_test_transform(mean=None, std=None, img_size=None):
    mean = IMAGENET_DEFAULT_MEAN if mean is None else mean
    std = IMAGENET_DEFAULT_STD if std is None else std

    primary_tfl = transforms.Resize((img_size, img_size))
    secondary_tfl = transforms.Compose([])
    final_tfl = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=torch.tensor(IMAGENET_DEFAULT_MEAN),
            std=torch.tensor(IMAGENET_DEFAULT_STD))
    ])
    return primary_tfl, secondary_tfl, final_tfl
