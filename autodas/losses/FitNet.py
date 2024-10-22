from __future__ import print_function

import torch.nn as nn


class HintLoss(nn.Module):

    def __init__(self):
        super(HintLoss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, input, target):
        return self.criterion(input, target)
