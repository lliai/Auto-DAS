'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''

import argparse
import os
import random
import sys
import time

import numpy as np
import torch

from diswotv2.datasets.imagenet import get_imagenet_dataloader
from diswotv2.helper.utils.latency_benchmark import get_model_latency
from diswotv2.losses import Similarity
from diswotv2.models.candidates.mutable import MasterNet
from diswotv2.models.candidates.mutable.searchspace.search_space_xxbl import \
    gen_search_space
from diswotv2.models.candidates.mutable.utils import (
    PlainNet, create_netblock_list_from_str)
from diswotv2.models.resnet_224 import resnet34
from diswotv2.predictor.zerocostproxy import (compute_gradnorm_score,
                                              compute_naswot_score,
                                              compute_NTK_score,
                                              compute_syncflow_score,
                                              compute_zen_score)
from diswotv2.searchspace.instincts import LinearInstinct
from diswotv2.searchspace.interactions import ParaInteraction
from diswotv2.utils.misc import mkfilepath


def parse_cmd_options(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument(
        '--zero_shot_score',
        type=str,
        default='Zen',
        help='could be: Zen (for Zen-NAS), TE (for TE-NAS)')
    parser.add_argument(
        '--evolution_max_iter',
        type=int,
        default=5000,  # int(48e4),
        help='max iterations of evolution.')
    parser.add_argument(
        '--budget_model_size',
        type=float,
        default=None,
        help='budget of model size, e.g., 1e6 means 1M params')
    parser.add_argument(
        '--budget_flops',
        type=float,
        default=None,
        help='budget of flops, e.g. , 1.8e6 means 1.8 GFLOPS')
    parser.add_argument(
        '--budget_latency',
        type=float,
        default=None,
        help='latency of forward inference, e.g., 1e-3 means 1ms.')

    parser.add_argument(
        '--max_layers',
        type=int,
        default=18,
        help='max number of layers of the network.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='number of instances in one mini-batch.')
    parser.add_argument(
        '--input_image_size',
        type=int,
        default=224,
        help='resolution of input image, usually 32 for CIFAR.')
    parser.add_argument(
        '--population_size',
        type=int,
        default=512,
        help='population size of evolution.')
    parser.add_argument(
        '--save_dir',
        type=str,
        default='./save_dirs/zen',
        help='output directory')
    parser.add_argument(
        '--gamma',
        type=float,
        default=1e-2,
        help='noise perturbation coefficient')
    parser.add_argument(
        '--num_classes', type=int, default=100, help='number of classes')
    module_opt, _ = parser.parse_known_args(argv)
    return module_opt


def get_new_random_structure_str(AnyPlainNet,
                                 structure_str,
                                 num_classes,
                                 get_search_space_func,
                                 num_replaces=1):
    """Generate random subnets based on source structure.

    Args:
        AnyPlainNet (_type_): _description_
        structure_str (_type_): _description_
        num_classes (_type_): _description_
        get_search_space_func (_type_): _description_
        num_replaces (int, optional): _description_. Defaults to 1.
    """
    the_net = AnyPlainNet(
        num_classes=num_classes, plainnet_struct=structure_str, no_create=True)
    assert isinstance(the_net, PlainNet)
    selected_random_id_set = set()
    for _ in range(num_replaces):
        random_id = random.randint(0, len(the_net.block_list) - 1)
        if random_id in selected_random_id_set:
            continue
        selected_random_id_set.add(random_id)
        # select a block randomly from block_list
        to_search_student_blocks_list_list = get_search_space_func(
            the_net.block_list, random_id)

        # generate a set of student blocks_list
        to_search_student_blocks_list = [
            x for sublist in to_search_student_blocks_list_list
            for x in sublist
        ]

        # choice one student structure randomly.
        new_student_block_str = random.choice(to_search_student_blocks_list)

        if len(new_student_block_str) > 0:
            new_student_block = create_netblock_list_from_str(
                new_student_block_str, no_create=True)
            assert len(new_student_block) == 1
            new_student_block = new_student_block[0]
            if random_id > 0:
                last_block_out_channels = the_net.block_list[random_id -
                                                             1].out_channels
                new_student_block.set_in_channels(last_block_out_channels)
            the_net.block_list[random_id] = new_student_block
        else:
            # replace with empty block
            the_net.block_list[random_id] = None
    pass  # end for

    # adjust channels and remove empty layer
    tmp_new_block_list = [x for x in the_net.block_list if x is not None]
    last_channels = the_net.block_list[0].out_channels
    for block in tmp_new_block_list[1:]:
        block.set_in_channels(last_channels)
        last_channels = block.out_channels
    the_net.block_list = tmp_new_block_list

    # get final random structure str
    new_random_structure_str = the_net.split(split_layer_threshold=6)
    return new_random_structure_str


def get_splitted_structure_str(AnyPlainNet, structure_str, num_classes):
    the_net = AnyPlainNet(
        num_classes=num_classes, plainnet_struct=structure_str, no_create=True)
    assert hasattr(the_net, 'split')
    return the_net.split(split_layer_threshold=6)


def get_latency(AnyPlainNet, random_structure_str, gpu, args):
    """get the latency constrain from random structure str."""
    the_model = AnyPlainNet(
        num_classes=args.num_classes,
        plainnet_struct=random_structure_str,
        no_create=False,
        no_reslink=False)
    if gpu is not None:
        the_model = the_model.cuda(gpu)
    the_latency = get_model_latency(
        model=the_model,
        batch_size=args.batch_size,
        resolution=args.input_image_size,
        in_channels=3,
        gpu=gpu,
        repeat_times=1,
        fp16=True)
    del the_model
    torch.cuda.empty_cache()
    return the_latency


def diswotv2_score(img, label, tmodel, smodel):
    """compute the fitness score of the model.

    Args:
        img (torch.Tensor): [B, C, H, W]
        label (torch.Tensor): [B]
        tmodel (torch.nn.Module): teacher model
        smodel (torch.nn.Module): student model
    """

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


def compute_nas_score(AnyPlainNet, random_structure_str, gpu, trainloader,
                      args):
    """Compute zero cost nas score.

    Args:
        AnyPlainNet (_type_): _description_
        random_structure_str (_type_): _description_
        gpu (_type_): _description_
        args (_type_): _description_
    """
    tnet = resnet34().cuda()

    # compute network zero-shot proxy score
    the_model = AnyPlainNet(
        num_classes=args.num_classes,
        plainnet_struct=random_structure_str,
        no_create=False,
        no_reslink=True)
    the_model = the_model.cuda(gpu)
    try:

        if args.zero_shot_score == 'Zen':
            the_nas_core_info = compute_zen_score(
                model=the_model,
                gpu=gpu,
                resolution=args.input_image_size,
                mixup_gamma=args.gamma,
                batch_size=args.batch_size,
                repeat=1)
            the_nas_score = the_nas_core_info['avg_nas_score']
        elif args.zero_shot_score == 'TE-NAS':
            the_nas_score = compute_NTK_score(
                model=the_model,
                gpu=gpu,
                resolution=args.input_image_size,
                batch_size=args.batch_size)

        elif args.zero_shot_score == 'Syncflow':
            the_nas_score = compute_syncflow_score(
                model=the_model,
                gpu=gpu,
                resolution=args.input_image_size,
                batch_size=args.batch_size)

        elif args.zero_shot_score == 'GradNorm':
            the_nas_score = compute_gradnorm_score(
                model=the_model,
                gpu=gpu,
                resolution=args.input_image_size,
                batch_size=args.batch_size)

        elif args.zero_shot_score == 'Flops':
            the_nas_score = the_model.get_FLOPs(args.input_image_size)

        elif args.zero_shot_score == 'Params':
            the_nas_score = the_model.get_model_size()

        elif args.zero_shot_score == 'Random':
            the_nas_score = np.random.randn()

        elif args.zero_shot_score == 'NASWOT':
            the_nas_score = compute_naswot_score(
                gpu=gpu,
                model=the_model,
                resolution=args.input_image_size,
                batch_size=args.batch_size)
        elif args.zero_shot_score == 'DISWOT':
            dataiter = iter(trainloader)
            img, label = next(dataiter)
            img = img.cuda()
            label = label.cuda()

            # snet = the_model
            the_model = the_model.cuda()
            criterion_sp = Similarity()
            # criterion_ickd = ICKDLoss()
            # criterion_ce = nn.CrossEntropyLoss()

            tfeature, tlogits = tnet.forward_with_features(img)
            sfeature, slogits = the_model(img, is_feat=True)

            # forward
            # criterion_ce(tlogits, label).backward()
            # criterion_ce(slogits, label).backward()

            # fc.weight.grad
            # import pdb; pdb.set_trace()
            # tcompressed = tnet.fc.weight.grad.cpu().unsqueeze(-1).unsqueeze(-1)
            # scompressed = the_model.fc_linear.netblock.weight.grad.cpu().unsqueeze(-1).unsqueeze(-1)
            score1 = -1 * criterion_sp(
                [tfeature[-2].cpu()],
                [sfeature[-2].cpu()])[0].detach().numpy()

            # import pdb; pdb.set_trace()
            # score2 = -1 * criterion_ickd([tcompressed], [scompressed])[0].detach().numpy()

            del dataiter
            del img
            del label
            del criterion_sp

            the_nas_score = score1[0]
        elif args.zero_shot_score == 'DISWOTv2':
            dataiter = iter(trainloader)
            img, label = next(dataiter)
            img = img.cuda()
            label = label.cuda()

            the_model = the_model.cuda()

            the_nas_score = diswotv2_score(img, label, tnet, the_model)

    except Exception as err:
        print(str(err))
        print('--- Failed structure: ')
        print(str(the_model))
        # raise err
        the_nas_score = -9999

    del the_model
    torch.cuda.empty_cache()
    return the_nas_score


def main(args, argv):
    gpu = args.gpu
    if gpu is not None:
        torch.cuda.set_device('cuda:{}'.format(gpu))
        torch.backends.cudnn.benchmark = True

    # best_structure_txt = os.path.join(args.save_dir, 'best_structure.txt')
    # if os.path.isfile(best_structure_txt):
    #     print('skip ' + best_structure_txt)
    #     return None

    trainloader, val_loader = get_imagenet_dataloader(
        dataset='imagenet', batch_size=32, num_workers=2, is_instance=False)

    # load masternet
    AnyPlainNet = MasterNet

    # source structure
    masternet = AnyPlainNet(
        num_classes=args.num_classes,
        opt=args,
        argv=argv,
        no_create=True,
        plainnet_struct=
        'SuperConvK7BNRELU(3,64,2,1)SuperResK3K3(64,64,2,64,2)SuperResK3K3(64,128,2,128,2)SuperResK3K3(128,256,2,256,2)SuperResK3K3(256,512,2,512,2)'
    )
    #  'SuperConvK3BNRELU(3,64,1,1)SuperResK1K5K1(64,168,1,16,3)SuperResK1K3K1(168,80,2,32,4)SuperResK1K5K1(80,112,2,16,3)SuperResK1K5K1(112,144,1,24,3)SuperResK1K3K1(144,32,2,40,1)SuperConvK1BNRELU(32,512,1,1)'
    initial_structure_str = str(masternet)

    popu_structure_list = []
    popu_zero_shot_score_list = []
    popu_modelsize_list = []

    start_timer = time.time()
    # main evolution search loop
    for loop_count in range(args.evolution_max_iter):
        # too many networks in the population pool, remove one with the smallest score
        while len(popu_structure_list) > args.population_size:
            min_zero_shot_score = min(popu_zero_shot_score_list)
            tmp_idx = popu_zero_shot_score_list.index(min_zero_shot_score)
            popu_zero_shot_score_list.pop(tmp_idx)
            popu_structure_list.pop(tmp_idx)
            popu_modelsize_list.pop(tmp_idx)

        if loop_count >= 1 and loop_count % 50 == 0:
            max_score = max(popu_zero_shot_score_list)
            min_score = min(popu_zero_shot_score_list)
            elasp_time = time.time() - start_timer
            print(
                f'loop_count={loop_count}/{args.evolution_max_iter}, max_score={max_score:4g}, min_score={min_score:4g}, time={elasp_time/3600:4g}h'
            )

        # ----- generate a random structure ----- #
        if len(popu_structure_list) <= 10:
            random_structure_str = get_new_random_structure_str(
                AnyPlainNet=AnyPlainNet,
                structure_str=initial_structure_str,
                num_classes=args.num_classes,
                get_search_space_func=gen_search_space,
                num_replaces=1)
        else:
            # popu structure is large enough
            tmp_idx = random.randint(0, len(popu_structure_list) - 1)
            tmp_random_structure_str = popu_structure_list[tmp_idx]
            # random select a structure from popu list
            random_structure_str = get_new_random_structure_str(
                AnyPlainNet=AnyPlainNet,
                structure_str=tmp_random_structure_str,
                num_classes=args.num_classes,
                get_search_space_func=gen_search_space,
                num_replaces=2)

        random_structure_str = get_splitted_structure_str(
            AnyPlainNet, random_structure_str, num_classes=args.num_classes)

        the_model = None

        if args.max_layers is not None:
            if the_model is None:
                the_model = AnyPlainNet(
                    num_classes=args.num_classes,
                    plainnet_struct=random_structure_str,
                    no_create=True,
                    no_reslink=False)
            the_layers = the_model.get_num_layers()
            if args.max_layers < the_layers:
                continue

        if args.budget_model_size is not None:
            # constraint the model size
            if the_model is None:
                the_model = AnyPlainNet(
                    num_classes=args.num_classes,
                    plainnet_struct=random_structure_str,
                    no_create=True,
                    no_reslink=False)
            the_model_size = the_model.get_model_size()
            if args.budget_model_size < the_model_size:
                continue

        if args.budget_flops is not None:
            # constraint the flops of model
            if the_model is None:
                the_model = AnyPlainNet(
                    num_classes=args.num_classes,
                    plainnet_struct=random_structure_str,
                    no_create=True,
                    no_reslink=False)
            the_model_flops = the_model.get_FLOPs(args.input_image_size)
            if args.budget_flops < the_model_flops:
                continue

        the_latency = np.inf
        if args.budget_latency is not None:
            # constraint the latency of model
            the_latency = get_latency(AnyPlainNet, random_structure_str, gpu,
                                      args)
            if args.budget_latency < the_latency:
                continue

        # compute fitness
        the_nas_score = compute_nas_score(AnyPlainNet, random_structure_str,
                                          gpu, trainloader, args)

        # add the selected subnet into population
        popu_structure_list.append(random_structure_str)
        popu_zero_shot_score_list.append(the_nas_score)
        popu_modelsize_list.append(the_model_size)

    return popu_structure_list, popu_zero_shot_score_list, popu_modelsize_list


if __name__ == '__main__':
    args = parse_cmd_options(sys.argv)

    info = main(args, sys.argv)
    if info is None:
        exit()

    popu_structure_list, popu_zero_shot_score_list, popu_modelsize_list = info

    # export best structure
    best_score = max(popu_zero_shot_score_list)
    best_idx = popu_zero_shot_score_list.index(best_score)
    best_structure_str = popu_structure_list[best_idx]
    the_model_size = popu_modelsize_list[best_idx]

    print(f'best_score={best_score:4g}')
    print(f'model_size={the_model_size:4g}')
    print(f'best_structure_str={best_structure_str}')

    best_structure_txt = os.path.join(args.save_dir, 'best_structure.txt')
    mkfilepath(best_structure_txt)
    with open(best_structure_txt, 'w') as fid:
        fid.write(best_structure_str)
