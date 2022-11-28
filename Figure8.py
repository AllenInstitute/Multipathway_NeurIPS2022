import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import os

from multichannel_net import MultipathwayNet
from analysis import MPNAnalysis


if __name__=='__main__':

    import argparse

    torch.manual_seed(539394)  

    plt.rc('font', size=20)
    plt.rcParams['figure.constrained_layout.use'] = True
    import matplotlib
    matplotlib.rcParams["mathtext.fontset"] = 'cm'
    
    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    nonlin = torch.nn.Tanh()

    depth_list = [2,3,4,7]

    fig_train, ax_train = plt.subplots(1,len(depth_list), figsize=(25,8))
    ax_train[0].set_ylabel('training error')
    
    fig_history = plt.figure(figsize=(24,10))
    gs = gridspec.GridSpec(2, 6,width_ratios=[2.2,1,1,2.2,1,1],figure=fig_history)

    timestep_list = [10000, 10000, 10000, 20000]

    min_val = 0.0
    max_val = 1.0

    mpna_list = []

    for di, depth in enumerate(depth_list):

        ax3d = fig_history.add_subplot(gs[di*3], projection='3d')

        mcn = MultipathwayNet(8,15, depth=depth, num_pathways=2, width=1000, bias=False, nonlinearity=nonlin)
        mpna = MPNAnalysis(mcn)
        mpna.train_mcn(timesteps=timestep_list[di], lr=0.01)

        ax_train[di].plot(mpna.loss_history)
        ax_train[di].set_xlabel('epoch')
        ax_train[di].set_title("$D={}$".format(depth))

        ax3d.set_title("$D={}$".format(depth))
        mpna.plot_K_history(ax3d, D=depth)

        mpna_list.append(mpna)

        K_list = [pathway[-1].to("cpu") for pathway in mpna.K_history]
        min_val_temp = np.min([torch.min(K) for K in K_list])
        max_val_temp = np.max([torch.max(K) for K in K_list])

        min_val = min(min_val_temp, min_val)
        max_val = max(max_val_temp, max_val)

    for di, seed in enumerate(depth_list):

        mpna = mpna_list[di]

        ax2 = fig_history.add_subplot(gs[di*3 +1])
        ax3 = fig_history.add_subplot(gs[di*3 +2])
        mpna.plot_K([ax2,ax3], labels=['a', 'b'], min_val=min_val, max_val=max_val)

    fig_train.suptitle("Training loss")
    fig_train.savefig('train_loss_figure_8.pdf')
    fig_history.savefig('Figure8.pdf')

    plt.show()

