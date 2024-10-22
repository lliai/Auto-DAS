import matplotlib.pyplot as plt
from matplotlib import ticker

from diswotv2.api.api import DisWOTAPI
from diswotv2.api.nas101_api import (NB101API, get_nb101_model_and_acc,
                                     get_nb101_teacher)
from diswotv2.datasets.cifar100 import get_cifar100_dataloaders
from diswotv2.models import resnet110
from diswotv2.models.candidates.mutable import mutable_resnet20
from diswotv2.searchspace.instincts import LinearInstinct
from diswotv2.searchspace.interactions import ParaInteraction
from diswotv2.utils.rank_consistency import spearman

# api = NB101API(
#     path='./data/nb101_kd_dict_9756ff660472a567ebabe535066c0e1f.pkl',
#     verbose=False)

api = NB101API(
    './data/nb101_dict_358bc32bd6537af8b13ed28d260e0c74.pkl', verbose=False)

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

train_loader, test_loader = get_cifar100_dataloaders(
    './data', batch_size=16, num_workers=0)

img, label = next(iter(train_loader))

img = img.cuda()

instinct = LinearInstinct()
instinct.update_genotype(
    'INPUT:(virtual_grad)UNARY:|abslog|sigmoid|normalized_sum|invert|')

interaction = ParaInteraction()
interaction.update_alleletype(
    "ALLELE# in:['k3']~       trans:['trans_drop', 'trans_multi_scale_r4', 'trans_pow2']~     weig:['w100_teacher_student']~  dist:['l1_loss']"
)

y_acc = []
z_zc = []
tm = get_nb101_teacher().cuda()

for i in range(200):
    # get the hash
    _hash = api.random_hash()

    sm, acc = get_nb101_model_and_acc(_hash)
    sm = sm.cuda()

    # compute score of the jointly
    score1 = interaction(img, label, tm, sm)
    score2 = instinct(img, label, sm)

    if acc < 0.7:
        continue

    y_acc.append(acc)
    z_zc.append(score1 + score2)

    del sm

# plot for correlation between accuracy and zero-cost proxy
print('spearman: ', spearman(y_acc, z_zc))
fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
plt.scatter(y_acc, z_zc)
# plt.colorbar()
plt.grid(lw=0.5, ls='-.')
plt.text(
    0.87,
    max(z_zc) * 0.98,
    f'Spearman={spearman(y_acc, z_zc):.2f}',
    style='italic')
plt.xlabel('Vanilla Acc (%)')
plt.ylabel('Auto-DAS Score')
# formatter = ticker.ScalarFormatter(useMathText=True)
# formatter.set_scientific(True)
# formatter.set_powerlimits((0, 0))
# ax.yaxis.set_major_formatter(formatter)
# plt.show()
plt.savefig('./corr_vanilla_nb101.png')
