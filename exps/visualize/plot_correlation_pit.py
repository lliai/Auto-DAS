import argparse
import gc

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import ticker
from torch.utils.data import DataLoader

from diswotv2.api.api import DisWOTAPI
from diswotv2.api.vit_api import DisWOT_API_VIT
from diswotv2.datasets.chaoyang import Chaoyang
from diswotv2.datasets.cifar100 import (get_cifar100_dataloaders,
                                        vit_cifar100_dataloaders)
from diswotv2.datasets.flowers import Flowers
from diswotv2.models import resnet56
from diswotv2.models.candidates.mutable.vit import PIT
from diswotv2.searchspace.instincts import LinearInstinct
from diswotv2.searchspace.interactions import ParaInteraction
from diswotv2.utils.rank_consistency import spearman

api = DisWOTAPI(
    './data/diswotv2_dict_4e1ea4a046d7e970b9151d8e23332ec5.pkl', verbose=True)

# plot the validity rate of the different search space (linear, tree, graph)
plt.rc('font', family='Times New Roman')
GLOBAL_DPI = 600
FIGSIZE = (8, 6)
PADINCHES = 0.1  # -0.005
GLOBAL_FONTSIZE = 34
GLOBAL_LABELSIZE = 30
GLOBAL_LEGENDSIZE = 25

font1 = {
    'family': 'Times New Roman',
    'weight': 'bold',
    'size': GLOBAL_LABELSIZE - 10
}

plt.rc('font', **font1)  # controls default text sizes
plt.rc('axes', titlesize=GLOBAL_LABELSIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=GLOBAL_LABELSIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=GLOBAL_LABELSIZE - 10)  # fontsize of the tick labels
plt.rc('ytick', labelsize=GLOBAL_LABELSIZE - 10)  # fontsize of the tick labels
plt.rc('legend', fontsize=GLOBAL_LEGENDSIZE - 10)  # legend fontsize
plt.rc('figure', titlesize=GLOBAL_LABELSIZE)

plt.rcParams['figure.constrained_layout.use'] = True

train_loader, val_loader = vit_cifar100_dataloaders(
            './data', 16)

img, label = next(iter(train_loader))

api = DisWOT_API_VIT('./data/diswotv2_pit.pth', mode='kd', dataset='c100')

instinct = LinearInstinct()
instinct.update_genotype(
    'INPUT:(grad)UNARY:|sigmoid|logsoftmax|abslog|no_op|')

interaction = ParaInteraction()
interaction.update_alleletype(
    "ALLELE# in:['k3']~       trans:['trans_sigmoid', 'trans_mask', 'trans_softmax_N']~       weig:['w1_teacher']~    dist:['kl_T1']"
)

y_acc = []
z_zc = []
tm = resnet56(num_classes=100)

# auto-das
tnet = resnet56(num_classes=100)
for struct, acc in iter(api):
    sm = PIT(struct, num_classes=100)

    score1 = interaction(img, label, tm, sm, interpolate=True)
    score2 = instinct(img, label, sm)

    if acc < 60:
        continue

    y_acc.append(acc)
    z_zc.append(score1+score2)

# plot for correlation between accuracy and zero-cost proxy
print('spearman: ', spearman(y_acc, z_zc))
fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
plt.scatter(y_acc, z_zc)
# plt.colorbar()
plt.grid(lw=0.5, ls='-.')
plt.text(
    65,
    max(z_zc) * 0.98,
    f'Spearman={spearman(y_acc, z_zc):.2f}',
    style='italic')
plt.xlabel('Distill Acc (%)')
plt.ylabel('Auto-DAS Score')
# formatter = ticker.ScalarFormatter(useMathText=True)
# formatter.set_scientific(True)
# formatter.set_powerlimits((0, 0))
# ax.yaxis.set_major_formatter(formatter)
plt.show()
