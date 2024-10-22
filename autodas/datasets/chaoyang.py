import json
import os

from PIL import Image

from .base import BaseDataset


class Chaoyang(BaseDataset):

    def __init__(self, data_path, split):
        super(Chaoyang, self).__init__(
            split, img_size=224, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        assert os.path.exists(data_path), "Data path '{}' not found".format(
            data_path)
        splits = ['train', 'test']
        assert split in splits, "Split '{}' not supported for Chaoyang".format(
            split)
        self.data_path = data_path
        with open(os.path.join(data_path, f'{split}.json'), 'r') as f:
            anns = json.load(f)
        self.data = anns

    def __len__(self):
        return len(self.data)

    def _get_data(self, index):
        ann = self.data[index]
        img = Image.open(os.path.join(self.data_path, ann['name']))
        return img, ann['label']
