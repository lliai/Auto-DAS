import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from matplotlib import ticker

from diswotv2.api.api import DisWOTAPI
from diswotv2.datasets.cifar100 import get_cifar100_dataloaders
from diswotv2.losses import ICKDLoss, Similarity
from diswotv2.models import resnet56, resnet110
from diswotv2.models.candidates.mutable import mutable_resnet20
from diswotv2.predictor.pruners import predictive
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

train_loader, test_loader = get_cifar100_dataloaders(
    './data', batch_size=16, num_workers=0)

img, label = next(iter(train_loader))

interaction = ParaInteraction()
interaction.update_alleletype(
    "ALLELE# in:['k2']~       trans:['trans_relu', 'trans_sigmoid', 'trans_abs']~     weig:['w100_teacher_student']~  dist:['kl_T8']"
)

instinct = LinearInstinct()
instinct.update_genotype(
    'INPUT:(grad)UNARY:|logsoftmax|no_op|frobenius_norm|normalized_sum|')

y_acc = []
z_zc = []
tm = resnet110(num_classes=100)

# diswotv2/auto-das
# for struct, acc in iter(api):
#     sm = mutable_resnet20(struct, num_classes=100)

#     # compute score of the jointly
#     score1 = interaction(img, label, tm, sm)
#     score2 = instinct(img, label, sm)

#     y_acc.append(acc)
#     z_zc.append(score1 + score2)

# nwot
# dataload_info = ['random', 3, 10]  # last one is num_classes
# for struct, acc in iter(api):
#     sm = mutable_resnet20(struct, num_classes=100)

#     score = predictive.find_measures(
#         sm,
#         train_loader,
#         dataload_info,
#         measure_names=['nwot'],
#         loss_fn=F.cross_entropy,
#         device=torch.device('cpu'))

#     y_acc.append(acc)
#     z_zc.append(score)

# diswotv1
tnet = resnet56(num_classes=100)
criterion_ickd = ICKDLoss()
criterion_sp = Similarity()
for struct, acc in iter(api):
    snet = mutable_resnet20(struct, num_classes=100)

    sfeature, slogits = snet(img, is_feat=True)
    tfeature, tlogits = tnet(img, is_feat=True)

    # score = -1 * criterion_ickd([sfeature[-1]],
    #                                 [sfeature[-1]])[0].detach().numpy()

    score = -1 * criterion_sp(tfeature[-1], sfeature[-1])[0].detach().numpy()

    y_acc.append(acc)
    z_zc.append(score)

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
plt.ylabel('DisWOT Score')
# formatter = ticker.ScalarFormatter(useMathText=True)
# formatter.set_scientific(True)
# formatter.set_powerlimits((0, 0))
# ax.yaxis.set_major_formatter(formatter)
plt.show()
