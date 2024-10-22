from abc import ABCMeta, abstractmethod

import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import create_test_transform, create_train_transform


class BaseDataset(Dataset, metaclass=ABCMeta):
    """
    Base class for dataset with support for offline distillation.
    """

    def __init__(self,
                 split,
                 offline=False,
                 feature_file=None,
                 mean=None,
                 std=None,
                 img_size=None,
                 strong_augmentation=False):
        if split == 'train':
            transforms = create_train_transform(
                mean=mean,
                std=std,
                img_size=img_size,
                offline=offline,
                strong_augmentation=strong_augmentation)
        else:
            transforms = create_test_transform(
                mean=mean, std=std, img_size=img_size)
        self.primary_tfl, self.secondary_tfl, self.final_tfl = transforms

        self.features = None
        if offline and split == 'train':
            assert feature_file is not None
            kd_data = np.load(feature_file)
            features = [
                kd_data[f'layer_{i}'] for i in range(len(kd_data.files))
            ]
            self.features = features

    @abstractmethod
    def _get_data(self, index):
        """
        Returns the image and its label at index.
        """
        pass

    def __getitem__(self, index):
        """ Returns the image and its label at index."""
        img, label = self._get_data(index)
        if self.features:
            features = [
                torch.from_numpy(f[index].copy()) for f in self.features
            ]
            for t in self.primary_tfl:
                img, features = t(img, features)
        else:
            img = self.primary_tfl(img)
            features = []

        img = self.secondary_tfl(img)
        img = self.final_tfl(img)

        return img, label, features
