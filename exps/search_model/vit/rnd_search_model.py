import argparse
import copy
import gc
import os
import random

import torch
import yaml
from torch.utils.data import DataLoader

from diswotv2.datasets.chaoyang import Chaoyang
from diswotv2.datasets.cifar10 import get_cifar10_dataloaders
from diswotv2.datasets.cifar100 import vit_cifar100_dataloaders
from diswotv2.datasets.flowers import Flowers
from diswotv2.models import resnet56
from diswotv2.models.candidates.mutable.vit.autoformer import AutoFormer
from diswotv2.models.candidates.mutable.vit.pit import PiT
from diswotv2.searchspace.instincts import LinearInstinct
from diswotv2.searchspace.interactions import ParaInteraction
from diswotv2.utils.misc import seed_all


def autoformer_configs(trial_num, num_classes=10):
    trial_configs = []
    choices = {
        'num_heads': [3, 4],
        'mlp_ratio': [3.5, 4.0],
        'hidden_dim': [192, 216, 240],
        'depth': [12, 13, 14]
    }
    dimensions = ['mlp_ratio', 'num_heads']

    for idx in range(trial_num):
        flag = False
        while not flag:
            depth = random.choice(choices['depth'])
            config = {
                dimension:
                [random.choice(choices[dimension]) for _ in range(depth)]
                for dimension in dimensions
            }
            config['hidden_dim'] = random.choice(choices['hidden_dim'])
            config['depth'] = depth

            temp_model = AutoFormer(
                arch_config=config, num_classes=num_classes)
            temp_params = sum(p.numel() for p in temp_model.parameters()
                              if p.requires_grad)

            if config not in trial_configs and args.af_low <= round(
                    temp_params / 1e6) <= args.af_up:
                flag = True
                trial_configs.append(config)
                print('generate {}-th AF config: {}, param: {} M'.format(
                    idx, config, round(temp_params / 1e6)))
            else:
                print('not suitable, AF param is:{} M'.format(
                    round(temp_params / 1e6)))

    return trial_configs


def pit_configs(trial_num, args, num_classes=10):

    trial_configs = []
    print('Pit param limit: {}--{}'.format(args.pit_low, args.pit_up))
    choices = {
        'base_dim': [16, 24, 32, 40],
        'mlp_ratio': [2, 4, 6, 8],
        'num_heads': [[2, 2, 2], [2, 2, 4], [2, 2, 8], [2, 4, 4], [2, 4, 8],
                      [2, 8, 8], [4, 4, 4], [4, 4, 8], [4, 8, 8], [8, 8, 8]],
        'depth': [[1, 6, 6], [1, 8, 4], [2, 4, 6], [2, 6, 4], [2, 6, 6],
                  [2, 8, 2], [2, 8, 4], [3, 4, 6], [3, 6, 4], [3, 8, 2]]
    }
    for idx in range(trial_num):
        flag = False
        while not flag:
            config = {}
            dimensions = ['mlp_ratio', 'num_heads', 'base_dim', 'depth']
            for dimension in dimensions:
                config[dimension] = random.choice(choices[dimension])
            temp_model = PiT(arch_config=config, num_classes=num_classes)
            temp_params = sum(p.numel() for p in temp_model.parameters()
                              if p.requires_grad)
            if config not in trial_configs and args.pit_low <= round(
                    temp_params / 1e6) <= args.pit_up:
                flag = True
                trial_configs.append(config)
                print('generate {}-th PIT config: {}, param: {} M'.format(
                    idx, config, round(temp_params / 1e6)))
            else:
                print('not suitable, PIT param is:{} M'.format(
                    round(temp_params / 1e6)))

    return trial_configs


def sample_trial_configs(model_type, args):
    pop = None
    trial_num = args.trial_num

    if args.dataset == 'c100':
        num_classes = 100
    elif args.dataset == 'flowers':
        num_classes = 102
    elif args.dataset == 'chaoyang':
        num_classes = 2

    if model_type == 'AutoFormerSub':
        pop = autoformer_configs(trial_num, num_classes=num_classes)
    elif model_type == 'PiT':
        pop = pit_configs(trial_num, args, num_classes=num_classes)
    return pop


def save_cfg(refer_path, searched_cfg, exp_name, cfg):
    with open(refer_path) as f:
        refer_data = yaml.safe_load(f)
    trial_data = copy.deepcopy(refer_data)

    if cfg.MODEL.TYPE == 'PiT':
        trial_data['PIT_SUBNET']['BASE_DIM'] = searched_cfg['base_dim']
        trial_data['PIT_SUBNET']['MLP_RATIO'] = searched_cfg['mlp_ratio']
        trial_data['PIT_SUBNET']['DEPTH'] = searched_cfg['depth']
        trial_data['PIT_SUBNET']['NUM_HEADS'] = searched_cfg['num_heads']

    elif cfg.MODEL.TYPE == 'AutoFormerSub':
        trial_data['AUTOFORMER_SUBNET']['HIDDEN_DIM'] = searched_cfg[
            'hidden_dim']
        trial_data['AUTOFORMER_SUBNET']['MLP_RATIO'] = searched_cfg[
            'mlp_ratio']
        trial_data['AUTOFORMER_SUBNET']['DEPTH'] = searched_cfg['depth']
        trial_data['AUTOFORMER_SUBNET']['NUM_HEADS'] = searched_cfg[
            'num_heads']

    yaml_dir = 'configs/auto/retrain/' + cfg.MODEL.TYPE
    if not os.path.exists(yaml_dir):
        os.makedirs(yaml_dir, exist_ok=True)

    with open(yaml_dir + '/{}.yaml'.format(exp_name), 'w') as f:
        yaml.safe_dump(trial_data, f, default_flow_style=False)


def standrd(data):
    min_value = torch.min(data)
    max_value = torch.max(data)
    res = (data - min_value) / (max_value - min_value)
    return res


def random_search(cfg, img, label, arch_pop, data_loader, num_classes):
    best_score = float('-inf')
    best_cfg = None
    tmodel = resnet56(num_classes=num_classes)

    for arch_idx, arch_cfg in enumerate(arch_pop):
        # build model
        smodel = AutoFormer(arch_config=arch_cfg, num_classes=num_classes)
        other_zc = compute_fitness_score(img, label, tmodel, smodel)

        print(f'config: {arch_cfg}')
        print(f'The {arch_idx}-th arch: Score: {other_zc}')
        if other_zc > best_score:
            best_score = other_zc
            best_cfg = arch_cfg

        gc.collect()
        torch.cuda.empty_cache()

    return best_cfg, best_score


def compute_fitness_score(img, label, tmodel, smodel):
    """ compute fitness score of a model"""
    instinct = LinearInstinct()
    instinct.update_genotype(
        'INPUT:(virtual_grad)UNARY:|abslog|sigmoid|normalized_sum|invert|')

    interaction = ParaInteraction()
    interaction.update_alleletype(
        "ALLELE# in:['k3']~       trans:['trans_drop', 'trans_multi_scale_r4', 'trans_pow2']~     weig:['w100_teacher_student']~  dist:['l1_loss']"
    )

    score1 = interaction(img, label, tmodel, smodel)
    score2 = instinct(img, label, smodel)
    return score1 + score2


def parse_args():
    parser = argparse.ArgumentParser(
        description='parameters',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # random
    parser.add_argument(
        '--trial_num', default=500, type=int, help='number of mutate')

    parser.add_argument(
        '--save_dir', default='work_dirs/search_model', type=str)

    parser.add_argument(
        '--refer_cfg',
        default='./configs/auto/autoformer/autoformer-ti-subnet_c100_base.yaml',
        type=str,
        help='save output path')

    # parser.add_argument(
    #     '--save_refer', default=None, type=str,
    #     help='save output path')

    parser.add_argument(
        '--other_zc', type=str, help='zero cost metric name', default=None)

    parser.add_argument(
        '--pit_up', default=22, type=float, help='pit param upper limit')
    parser.add_argument(
        '--pit_low', default=4, type=float, help='pit param lower limit')

    parser.add_argument(
        '--af_up', default=9, type=float, help='autoformer param upper limit')
    parser.add_argument(
        '--af_low', default=4, type=float, help='autoformer param lower limit')

    parser.add_argument('--iterations', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save', type=str, default='./output')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--scale', type=bool, default=False)
    parser.add_argument('--sample_num', type=int, default=50)
    parser.add_argument('--dataset', type=str, default='c100')

    parser.add_argument('--search_space', type=str, default='autoformer')

    return parser.parse_args()


if __name__ == '__main__':
    # add argparse to support iteration
    args = parse_args()

    if args.dataset == 'c100':
        num_classes = 100
    elif args.dataset == 'flowers':
        num_classes = 102
    elif args.dataset == 'chaoyang':
        num_classes = 2

    # seed all
    seed_all(args.seed)

    # build dataloader
    if args.dataset == 'c100':
        # build dataloader
        train_loader, val_loader = vit_cifar100_dataloaders(
            args.data_path, args.batch_size)
    elif args.dataset == 'flower':
        train_dataset = Flowers(args.data_path, split='train')
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4)
    elif args.dataset == 'chaoyang':
        train_dataset = Chaoyang(args.data_path, split='train')
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4)
    else:
        raise NotImplementedError

    print('dataloader built')

    # get img, label
    img, label = next(iter(train_loader))
    if torch.cuda.is_available():
        img = img.cuda()
        label = label.cuda()

    # random search
    arch_pop = sample_trial_configs(args.search_space, args)

    best_cfg, best_score = random_search(args, img, label, arch_pop,
                                         train_loader, num_classes)
