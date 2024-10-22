# plot the evolution process of the ranking consistency
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

plt.rc('font', family='Times New Roman')
GLOBAL_DPI = 300
FIGSIZE = (8, 6)
PADINCHES = 0.1  # -0.005
GLOBAL_FONTSIZE = 29
GLOBAL_LABELSIZE = 32
GLOBAL_LEGENDSIZE = 20

font1 = {
    'family': 'Times New Roman',
    # 'weight': 'bold',
    'size': GLOBAL_LABELSIZE
}

plt.rc('font', **font1)  # controls default text sizes
plt.rc('axes', titlesize=GLOBAL_LABELSIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=GLOBAL_LABELSIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=GLOBAL_LABELSIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=GLOBAL_LABELSIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=GLOBAL_LEGENDSIZE)  # legend fontsize
plt.rc('figure', titlesize=GLOBAL_LABELSIZE)


def plot_evolving_vs_random_search(evolv_csv_path, random_csv_path,
                                   aes_csv_path, fig_path):
    # fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    # fig.add_subplot(111, frameon=False)

    # extract the iters, and the spearman from the csv file
    plt.figure(figsize=FIGSIZE, dpi=GLOBAL_DPI)
    iters = []
    spearman = []
    with open(evolv_csv_path, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                continue
            iters.append(int(row[0]))
            spearman.append(float(row[1]))

    # plt.tick_params(
    #     labelcolor='none', top=False, bottom=False, left=False, right=False)

    plt.grid(linestyle='-.', lw=2, alpha=0.9)
    # xticks = np.arange(0, 1000, 250)
    plt.tick_params(labelsize=GLOBAL_LEGENDSIZE)

    plt.plot(
        iters,
        spearman,
        label='Evolution w/o AES',
        lw=4,
        linestyle='-',
        color='#F8AC8C')

    # extract the aes evolution infos
    iters = []
    spearman = []
    with open(aes_csv_path, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                continue
            iters.append(int(row[0]))
            spearman.append(float(row[1]))

    plt.plot(
        iters,
        spearman,
        label='Evolution w/ AES',
        lw=4,
        linestyle='-',
        color='#C82423')

    # extract the iters, and the spearman from the csv file
    iters = []
    spearman = []
    with open(random_csv_path, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                continue
            iters.append(int(row[0]))
            spearman.append(float(row[1]))

    plt.plot(
        iters,
        spearman,
        label='Random Search',
        lw=4,
        linestyle='-',
        color='#2878B5')

    plt.xlim(0, 1000)
    plt.ylim(0.5, 0.85)

    # plt.title('Evolution Process of the Ranking Consistency')
    plt.xlabel('Iteration', fontsize=GLOBAL_FONTSIZE)
    plt.ylabel('Fitness', fontsize=GLOBAL_FONTSIZE)
    plt.legend(loc='lower right', fontsize=GLOBAL_LEGENDSIZE)

    plt.tight_layout()
    plt.savefig(fig_path)


if __name__ == '__main__':
    evolv_csv_path = './output/rnd_search_interaction_666_2023_03_23.csv'
    random_csv_path = './output/rnd_search_interaction_555_2023_03_23.csv'
    aes_csv_path = './output/random_jointly_LinearInstinct_1000_129_spearman_rho.csv'
    fig_path = './evolution_vs_random_resnet18_tree_structure.png'
    plot_evolving_vs_random_search(evolv_csv_path, random_csv_path,
                                   aes_csv_path, fig_path)

    
    print('Done!')