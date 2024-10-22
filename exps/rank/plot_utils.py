import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import ticker

GLOBAL_DPI = 600
FIGSIZE = (8, 6)
PADINCHES = 0.1  # -0.005
GLOBAL_FONTSIZE = 34
GLOBAL_LABELSIZE = 30
GLOBAL_LEGENDSIZE = 20


def setup_plt():
    plt.rc('font', family='Times New Roman')

    font1 = {
        'family': 'Times New Roman',
        'weight': 'bold',
        'size': GLOBAL_LABELSIZE
    }

    plt.rc('font', **font1)  # controls default text sizes
    plt.rc('axes', titlesize=GLOBAL_LABELSIZE)  # fontsize of the axes title
    # fontsize of the x and y labels
    plt.rc('axes', labelsize=GLOBAL_LABELSIZE)
    # fontsize of the tick labels
    plt.rc('xtick', labelsize=GLOBAL_LABELSIZE - 10)
    # fontsize of the tick labels
    plt.rc('ytick', labelsize=GLOBAL_LABELSIZE - 10)
    plt.rc('legend', fontsize=GLOBAL_LEGENDSIZE)  # legend fontsize
    plt.rc('figure', titlesize=GLOBAL_LABELSIZE)

    ax = plt.gca()
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
