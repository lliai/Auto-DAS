# v3 means we only adopt unary op to monitor the EZNAS.
import argparse
import csv
import logging
import random

import matplotlib.pyplot as plt
import torch

from diswotv2.helper.distiller import diswotv2Distiller
from diswotv2.searchspace.interactions import ParaInteraction
from diswotv2.utils.misc import seed_all

logger = logging.getLogger('evolution_search_zc')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh = logging.FileHandler('evolution_search_zc.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(logging.StreamHandler())


def all_same(items):
    return all(x == items[0] for x in items)


def random_search(arch, iterations=1000, distiller=None):
    # prepare data for matplotlib plot
    idx = []
    rks = []

    best_zc = -1e8
    best_struct = ParaInteraction()

    # run the cycle
    logger.info('Begin the random search process...')
    for i in range(iterations):
        struct = ParaInteraction()
        score = distiller.estimate_rewards(struct)

        if best_zc < score:
            best_zc = score
            best_struct = struct

        logger.info(f'Iter: {i} Best SP: {best_struct}')

        # add data for matplotlib plot
        idx.append(i)
        rks.append(best_zc)

    # plot the evolution process
    save_name = f'evolution_{arch}_{type(best_struct).__name__}_{iterations}_{random.randint(0, 1000)}'
    plt.plot(idx, rks)
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.savefig(f'./output/rnd_search_diswotv2_zc/{save_name}.png')

    # save idx and rks into csv file
    with open(f'./output/rnd_search_diswotv2_zc/{save_name}.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(zip(idx, rks))

    return best_struct


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='running parameters',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # general parameters for data and qnn
    parser.add_argument(
        '--seed',
        default=42,
        type=int,
        help='random seed for results reproduction')
    parser.add_argument(
        '--arch',
        default='resnet18',
        type=str,
        help='dataset name',
        choices=['resnet18', 'mobilenetv2'])
    parser.add_argument(
        '--dataset',
        type=str,
        default='cifar100',
        choices=['cifar100'],
        help='dataset')

    parser.add_argument(
        '--batch_size',
        default=4,
        type=int,
        help='mini-batch size for data loader')
    parser.add_argument(
        '--workers',
        default=0,
        type=int,
        help='number of workers for data loader')
    parser.add_argument(
        '--data_path',
        default='E:/all_imagenet_data',
        type=str,
        help='path to ImageNet data')
    parser.add_argument(
        '--model_s',
        type=str,
        default='resnet20',
        choices=[
            'resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44',
            'resnet56', 'resnet110', 'resnet8x4', 'resnet32x4', 'wrn_16_1',
            'wrn_16_2', 'wrn_40_1', 'wrn_40_2', 'vgg8', 'vgg11', 'vgg13',
            'vgg16', 'vgg19', 'ResNet50', 'MobileNetV2', 'ShuffleV1',
            'ShuffleV2'
        ])
    parser.add_argument(
        '--path_t',
        type=str,
        default='./save/models/resnet110_vanilla/ckpt_epoch_240.pth',
        help='teacher model snapshot')
    parser.add_argument(
        '--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument(
        '--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument(
        '--lr_decay_epochs',
        type=str,
        default='10,20',
        help='where to decay lr, can be a list')

    # quantization parameters
    parser.add_argument(
        '--n_bits_w',
        default=2,
        type=int,
        help='bitwidth for weight quantization')
    parser.add_argument(
        '--channel_wise',
        action='store_true',
        help='apply channel_wise quantization for weights')
    parser.add_argument(
        '--n_bits_a',
        default=8,
        type=int,
        help='bitwidth for activation quantization')
    parser.add_argument(
        '--act_quant',
        action='store_true',
        help='apply activation quantization')
    parser.add_argument('--disable_8bit_head_stem', action='store_true')
    parser.add_argument('--test_before_calibration', action='store_true')
    parser.add_argument(
        '-r',
        '--gamma',
        type=float,
        default=1,
        help='weight for classification')
    parser.add_argument(
        '-a', '--alpha', type=float, default=1, help='weight balance for KD')
    parser.add_argument(
        '-b',
        '--beta',
        type=float,
        default=1,
        help='weight balance for other losses')
    # weight calibration parameters
    parser.add_argument(
        '--num_samples',
        default=1024,
        type=int,
        help='size of the calibration dataset')

    parser.add_argument(
        '--weight',
        default=0.01,
        type=float,
        help='weight of rounding cost vs the reconstruction loss.')
    parser.add_argument(
        '--sym',
        action='store_true',
        help='symmetric reconstruction, not recommended')
    parser.add_argument(
        '--b_start',
        default=20,
        type=int,
        help='temperature at the beginning of calibration')
    parser.add_argument(
        '--b_end',
        default=2,
        type=int,
        help='temperature at the end of calibration')
    parser.add_argument(
        '--warmup',
        default=0.2,
        type=float,
        help='in the warmup period no regularization is applied')
    parser.add_argument(
        '--step', default=20, type=int, help='record snn output per step')
    parser.add_argument(
        '--use_bias',
        action='store_true',
        help='fix weight bias and variance after quantization')
    parser.add_argument(
        '--vcorr', action='store_true', help='use variance correction')
    parser.add_argument(
        '--bcorr', action='store_true', help='use bias correction')

    # activation calibration parameters
    parser.add_argument(
        '--iterations',
        default=100,
        type=int,
        help='number of iteration for LSQ')
    parser.add_argument(
        '--lr', default=4e-4, type=float, help='learning rate for LSQ')
    parser.add_argument(
        '--p', default=2.4, type=float, help='L_p norm minimization for LSQ')

    # search space structure
    parser.add_argument(
        '--search_structure',
        default='tree',
        type=str,
        help='search space structure, linear, tree or graph',
    )

    # popu size
    parser.add_argument(
        '--popu_size',
        default=10,
        type=int,
        help='population size should be larger than 10',
    )

    # log_folder
    parser.add_argument(
        '--log_folder',
        default='./output/rnd_search_diswotv2_zc',
        type=str,
        help='path to ImageNet data')

    args = parser.parse_args()

    seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    # process iterations
    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = []
    args.lr_decay_epochs.extend(int(it) for it in iterations)

    # build kd-zero distiller
    distiller = diswotv2Distiller(args)

    logger.info('Begin Random Search...')
    random_search(args.arch, args.iterations, distiller)
