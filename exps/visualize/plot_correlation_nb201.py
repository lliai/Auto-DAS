import matplotlib.pyplot as plt

from diswotv2.api.nas201_api import (NB201KDAPI, get_network_by_index,
                                     get_teacher_best_model,
                                     random_sample_and_get_gt)
from diswotv2.datasets.cifar100 import get_cifar100_dataloaders
from diswotv2.searchspace.instincts import LinearInstinct
from diswotv2.searchspace.interactions import ParaInteraction
from diswotv2.utils.rank_consistency import spearman

# api = NB201KDAPI('./data/nb201_kd_dict_1dd544f95b3094a251a0815d3a616dff.pkl')
api = None

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
instinct.update_genotype('INPUT:(grad)UNARY:|logsoftmax|no_op|tanh|l1_norm|')

interaction = ParaInteraction()
interaction.update_alleletype(
    "ALLELE# in:['k2']~       trans:['trans_mish', 'trans_pow2', 'trans_softmax_N']~  weig:['w100_teacher_student']~  dist:['multiply']"
)

y_acc = []
z_zc = []
tm = get_teacher_best_model()
tm = tm.cuda()

for i in range(200):
    # get the model
    if api is None:
        sm, acc = random_sample_and_get_gt()
    else:
        rnd_idx = api.random_idx()
        acc = api.query_by_idx(rnd_idx)  # kd acc
        sm, _ = get_network_by_index(int(rnd_idx))

    sm = sm.cuda()

    # compute score of the jointly
    score1 = interaction(img, label, tm, sm)
    score2 = instinct(img, label, sm)

    if acc < 30:
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
    min(y_acc) * 1.01,
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
plt.savefig('./corr_vanilla_nb201.png')
