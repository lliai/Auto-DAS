import argparse

import torch

from diswotv2.api.api import DisWOTAPI
from diswotv2.datasets.cifar100 import get_cifar100_dataloaders
from diswotv2.helper.utils.flop_benchmark import get_model_infos
from diswotv2.models import resnet56
from diswotv2.models.candidates.mutable import mutable_resnet20
from diswotv2.searchspace.instincts import LinearInstinct
from diswotv2.searchspace.interactions import ParaInteraction
from diswotv2.utils.misc import seed_all


def compute_fitness_score(img, label, tmodel, smodel):
    """ compute fitness score of a model"""

    interaction = ParaInteraction()
    interaction.update_alleletype(
        "ALLELE# in:['k2']~       trans:['trans_relu', 'trans_sigmoid', 'trans_abs']~     weig:['w100_teacher_student']~  dist:['kl_T8']"
    )

    instinct = LinearInstinct()
    instinct.update_genotype(
        'INPUT:(grad)UNARY:|logsoftmax|no_op|frobenius_norm|normalized_sum|')

    # compute score of the jointly
    score1 = interaction(img, label, tmodel, smodel)
    score2 = instinct(img, label, smodel)

    return score1 + score2


def random_search(img, label, api, param_constraint=100):
    """ random search for the best model with highest fitness score"""
    best_struct = None
    best_acc = 0
    best_score = 0

    teacher = resnet56()
    if torch.cuda.is_available():
        teacher = teacher.cuda()

    for i, (struct, acc) in enumerate(iter(api)):
        model = mutable_resnet20(struct, num_classes=100)
        flops, param = get_model_infos(model, shape=(1, 3, 32, 32))
        if param > param_constraint:
            continue

        if torch.cuda.is_available():
            model = model.cuda()

        score = compute_fitness_score(img, label, teacher, model)

        if score > best_score:
            best_acc = acc
            best_struct = struct
            best_score = score

        print(
            f' * iteration {i}: struct: {best_struct}, gt acc: {acc}, best: {score}'
        )
    return best_struct, best_acc, best_score


if __name__ == '__main__':
    # add argparse to support iteration
    parser = argparse.ArgumentParser('random search jointly')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save', type=str, default='./output')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--scale', type=bool, default=False)
    parser.add_argument('--sample_num', type=int, default=50)
    parser.add_argument('--param_constraint', type=float, default=2)

    # benchmark file
    parser.add_argument(
        '--benchmark',
        type=str,
        default='./data/nb101_dict_358bc32bd6537af8b13ed28d260e0c74.pkl')

    args = parser.parse_args()

    # seed all
    seed_all(args.seed)

    # build dataloader
    train_loader, val_loader = get_cifar100_dataloaders(
        args.data_path, args.batch_size)
    print('dataloader built')

    # get img, label
    img, label = next(iter(train_loader))
    if torch.cuda.is_available():
        img = img.cuda()
        label = label.cuda()

    # build api
    api = DisWOTAPI(
        './data/diswotv2_dict_4e1ea4a046d7e970b9151d8e23332ec5.pkl',
        verbose=True)
    # random search
    best_struct, best_acc, best_score = random_search(img, label, api,
                                                      args.param_constraint)
    print(
        f'best struct: {best_struct}, best acc: {best_acc}, best score: {best_score}'
    )
