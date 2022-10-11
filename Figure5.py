import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import os

from multichannel_net import MultipathwayNet
from analysis import MPNAnalysis


if __name__=='__main__':

    import argparse

    torch.manual_seed(576)  # 87578)  #576)  # 345345)

    plt.rc('font', size=20)
    plt.rcParams['figure.constrained_layout.use'] = True
    import matplotlib
    matplotlib.rcParams["mathtext.fontset"] = 'cm'
    
    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    nonlin = None

    depth_list = [2,3,4,7]

    fig_train, ax_train = plt.subplots(1,len(depth_list), figsize=(25,8))
    ax_train[0].set_ylabel('training error')
    
    fig_history = plt.figure(figsize=(24,10))
    gs = gridspec.GridSpec(2, 6,width_ratios=[2.2,1,1,2.2,1,1],figure=fig_history)

    timestep_list = [1000, 1000, 1400, 10000]
    timestep_list = [1000, 10000, 10000, 20000]

    for di, depth in enumerate(depth_list):

        ax3d = fig_history.add_subplot(gs[di*3], projection='3d')
        ax2 = fig_history.add_subplot(gs[di*3 +1])
        ax3 = fig_history.add_subplot(gs[di*3 +2])

        mcn = MultipathwayNet(8,15, depth=depth, num_pathways=2, width=1000, bias=False, nonlinearity=nonlin)
        mpna = MPNAnalysis(mcn)
        mpna.train_mcn(timesteps=timestep_list[di], lr=0.01)

        ax_train[di].plot(mpna.loss_history)
        ax_train[di].set_xlabel('epoch')
        ax_train[di].set_title("$D={}$".format(depth))

        ax3d.set_title("$D={}$".format(depth))
        mpna.plot_K([ax2,ax3], labels=['a', 'b'])
        mpna.plot_K_history(ax3d, D=depth)

    fig_train.suptitle("Training loss")
    fig_train.savefig('train_loss_figure_5.pdf')
    fig_history.savefig('Figure5.pdf')

    plt.show()

