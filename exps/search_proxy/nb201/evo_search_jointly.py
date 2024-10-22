import argparse
import csv
import gc
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from diswotv2.api.nas201_api import (get_teacher_best_model,
                                     random_sample_and_get_gt)
from diswotv2.datasets.cifar100 import get_cifar100_dataloaders
from diswotv2.searchspace.instincts import LinearInstinct
from diswotv2.searchspace.interactions import ParaInteraction
from diswotv2.utils.misc import all_same, is_anomaly, seed_all
from diswotv2.utils.rank_consistency import kendalltau, pearson, spearman


def compute_rank_consistency(img: Tensor,
                             label: Tensor,
                             instinct: LinearInstinct,
                             interaction: ParaInteraction,
                             scale: bool = False,
                             sample_num: int = 50):
    """Compute rank consistency of the search space.

    Args:
        img (Tensor): input image
        label (Tensor): ground truth label
        instinct (LinearInstinct): instinct
        interaction (ParaInteraction): interaction
        scale (bool, optional): scale the scores. Defaults to False.
        sample_num (int, optional): number of samples. Defaults to 50.

    Returns:
        float: kendalltau
        float: spearman
        float: pearson
    """
    # check memory whether have the rank info.
    if instinct.reward_score is not None:
        return instinct.reward_score
    if interaction.reward_score is not None:
        return interaction.reward_score

    gt_list = []
    zc1_list = []  # instinct
    zc2_list = []  # interaction

    tm = get_teacher_best_model()

    # to cuda
    if torch.cuda.is_available():
        tm = tm.cuda()

    # traverse the search space in api
    for i in range(sample_num):
        # get the model
        sm, acc = random_sample_and_get_gt()
        # to cuda
        if torch.cuda.is_available():
            sm = sm.cuda()

        # compute score of the jointly
        score1 = interaction(img, label, tm, sm)
        score2 = instinct(img, label, sm)

        # exception handling
        if score1 == -1 or score2 == -1:
            return -1, -1, -1

        if np.isnan(score1) or np.isnan(score2):
            return -1, -1, -1

        # early stop
        if len(zc1_list) > 3 and all_same(zc1_list):
            return -1, -1, -1
        if len(zc2_list) > 3 and all_same(zc2_list):
            return -1, -1, -1

        # record
        gt_list.append(acc)
        zc1_list.append(score1)
        zc2_list.append(score2)

        # gabage collection
        del sm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # gabage collection
    del tm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # scale two score list to [0, 1] and then add them .
    if scale:
        zc1_list = (zc1_list - np.min(zc1_list)) / \
            (np.max(zc1_list) - np.min(zc1_list))
        zc2_list = (zc2_list - np.min(zc2_list)) / \
            (np.max(zc2_list) - np.min(zc2_list))
    zc_list = [x + y for x, y in zip(zc1_list, zc2_list)]

    # compute rank consistency
    kd = kendalltau(gt_list, zc_list)
    sp = spearman(gt_list, zc_list)
    ps = pearson(gt_list, zc_list)

    if np.isnan(sp):
        return -1, -1, -1
    else:
        instinct.reward_score = [kd, sp, ps]
        interaction.reward_score = [kd, sp, ps]

    return kd, sp, ps


def evolution_search_jointly(img: Tensor,
                             label: Tensor,
                             iterations: int = 1000,
                             popu_size: int = 50,
                             prob_mutate: float = 0.1,
                             prob_crossover: float = 0.5,
                             topk: float = 0.2,
                             scale: bool = False,
                             sample_num: int = 50):
    """ Evolution search algorithm for jointly search instinct and interaction.

    Args:
        img (Tensor): input image
        label (Tensor): input label
        iterations (int, optional): number of iterations. Defaults to 1000.
        logger ([type], optional): logger. Defaults to None.
        popu_size (int, optional): population size. Defaults to 50.
        prob_mutate (float, optional): probability of mutation. Defaults to 0.1.
        prob_crossover (float, optional): probability of crossover. Defaults to 0.5.
        topk (float, optional): topk. Defaults to 0.2.
        scale (bool, optional): whether to scale the scores. Defaults to False.
        sample_num (int, optional): number of sampling. Defaults to 50.

    """

    # random initialize `popu_size` individuals for evolution search
    population = []
    print(f'random initialize {popu_size} individuals')

    while len(population) < popu_size:
        instinct = LinearInstinct()
        interaction = ParaInteraction()
        kd, score, pr = compute_rank_consistency(img, label, instinct,
                                                 interaction, scale,
                                                 sample_num)

        if is_anomaly(score):
            continue

        print(
            f'random initialize {len(population)}th individual Got kd: {kd} sp: {score} pr: {pr}'
        )
        population.append((instinct, interaction))

    # prepare data for evolution search illustration.
    csv_save_data = []

    # run the evolution search algorithm
    print(f'run the evolution search algorithm for {iterations} iterations')

    for i in range(iterations):
        # scores = [
        #     compute_rank_consistency(img, label, instinct, interaction, scale,
        #                              sample_num)[0]
        #     for instinct, interaction in population
        # ]

        all_score_list = [
            compute_rank_consistency(img, label, instinct, interaction, scale,
                                     sample_num)
            for instinct, interaction in population
        ]

        # kd scores
        kd_scores = [kd for kd, sp, ps in all_score_list]
        # sp scores
        sp_scores = [sp for kd, sp, ps in all_score_list]
        # ps scores
        ps_scores = [ps for kd, sp, ps in all_score_list]

        scores = sp_scores  # use sp score as the reward

        # select the best one from the population
        scores = np.array(scores)
        argidxs = np.argsort(scores)[::-1]

        # best structure on the run
        running_struct = population[argidxs[0]]
        print(
            f'Iter: {i} Best RK: {scores[argidxs[0]]} Best Instinct: {running_struct[0]} Best Interaction: {running_struct[1]}'
        )

        # record contents for csv file
        csv_save_data.append({
            'iteration': i,
            'kendall_tau': kd_scores[argidxs[0]],
            'spearman_rho': sp_scores[argidxs[0]],
            'pearson_rho': ps_scores[argidxs[0]],
        })

        # sample candidates for crossover and mutation
        candidates = [population[i]
                      for i in argidxs[:int(popu_size * topk)]]  # TopN
        offspring_struct = random.choice(candidates)
        best_struct2 = random.choice(candidates)

        if np.random.rand() < prob_crossover:
            # 1. cross over
            if np.random.rand() < 0.5:
                offspring_struct[0] = best_struct2[0]
            else:
                offspring_struct[1] = best_struct2[1]

        if np.random.rand() < prob_mutate:
            # 2. mutation
            if np.random.rand() < 0.5:
                offspring_struct[0] = offspring_struct[0].mutate()
            else:
                offspring_struct[1] = offspring_struct[1].mutate()

        # 3. Diversity-prompting selection
        offspring_score = compute_rank_consistency(img, label,
                                                   offspring_struct[0],
                                                   offspring_struct[1], scale,
                                                   sample_num)[0]
        newbie = (LinearInstinct(), ParaInteraction())
        newbie_score = compute_rank_consistency(img, label, newbie[0],
                                                newbie[1], scale,
                                                sample_num)[0]

        # 4. delete the deteriorated individual
        del population[argidxs[-1]]

        # 5. add better offspring or newbie to population.
        if offspring_score > newbie_score:
            population.append(offspring_struct)
        else:
            population.append(newbie)

        # 6. assert the population size is `popu_size`
        assert len(
            population
        ) == popu_size, f'population size should be: {len(population)}'

    # evaluate the fitness of all structures
    scores = [
        compute_rank_consistency(img, label, s[0], s[1], scale, sample_num)[0]
        for s in population
    ]
    argidxs = np.argsort(scores)[::-1]
    running_struct = population[argidxs[0]]
    print(
        f'After {iterations} iters: Best SP:{scores[argidxs[0]]} Best Instinct: {running_struct[0]} Best Interaction: {running_struct[1]}'
    )

    # plot the evolution process
    save_name = f'evolution_jointly_{type(newbie).__name__}_{iterations}_{popu_size}_{random.randint(0, 1000)}'

    # plot the evolution process
    idxs, rk_kd, rk_sp, rk_ps = [], [], [], []
    for _dict in csv_save_data:
        idxs.append(_dict['iteration'])
        rk_kd.append(_dict['kendall_tau'])
        rk_sp.append(_dict['spearman_rho'])
        rk_ps.append(_dict['pearson_rho'])

    post_process(idxs, rk_kd, save_name, 'kendall_tau')
    post_process(idxs, rk_sp, save_name, 'spearman_rho')
    post_process(idxs, rk_ps, save_name, 'pearson_rho')

    return running_struct[0], running_struct[1], scores[argidxs[0]]


def post_process(idxs, rks, save_name, rank_name):
    plt.grid()
    # rank correlation
    plt.plot(idxs, rks)
    plt.xlabel('Iteration')
    plt.ylabel('kendall tau')
    plt.savefig(f'./output/evo_search_jointly/{save_name}_{rank_name}.png')

    # save idx and rks into csv file
    with open(f'./output/evo_search_jointly/{save_name}_{rank_name}.csv',
              'w') as f:
        writer = csv.writer(f)
        writer.writerows(zip(idxs, rks))


if __name__ == '__main__':
    # add argparse to support iteration
    parser = argparse.ArgumentParser('evolution search jointly')
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save', type=str, default='./output')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--scale', type=bool, default=False)

    # for evolution search algorithm
    parser.add_argument('--popu_size', type=int, default=50)
    parser.add_argument('--prob_mutate', type=float, default=0.1)
    parser.add_argument('--prob_crossover', type=float, default=0.5)
    parser.add_argument('--topk', type=float, default=0.2)

    # sample_num for rank consistency
    parser.add_argument('--sample_num', type=int, default=50)

    args = parser.parse_args()

    # seed all
    seed_all(args.seed)

    # build dataloader
    train_loader, val_loader = get_cifar100_dataloaders(
        args.data_path, args.batch_size)

    # get img, label
    img, label = next(iter(train_loader))

    # to cuda
    if torch.cuda.is_available():
        img = img.cuda()
        label = label.cuda()

    # evolution search
    print('evolution search jointly begin...')
    best_interaction, best_instinct, best_rk = evolution_search_jointly(
        img, label, args.iterations, args.popu_size, args.prob_mutate,
        args.prob_crossover, args.topk, args.scale, args.sample_num)

    print('evolution search jointly end...')
    print(f' * best_rk: {best_rk:.4f}')
    print(f' * best_interaction: {best_interaction}')
    print(f' * best_instinct: {best_instinct}')
