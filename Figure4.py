import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import os

from multichannel_net import MultipathwayNet
from analysis import MPNAnalysis


if __name__=='__main__':

    import argparse

    torch.manual_seed(576)

    plt.rc('font', size=20)
    plt.rcParams['figure.constrained_layout.use'] = True
    import matplotlib
    matplotlib.rcParams["mathtext.fontset"] = 'cm'
    
    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    nonlin = None

    depth = 2

    num_checkpoints = 10
    fig, ax = plt.subplots(1,num_checkpoints, figsize=(15,3))

    timesteps = 1

    mcn = MultipathwayNet(8,15, depth=depth, num_pathways=2, width=1000, bias=False, nonlinearity=nonlin)
    mpna = MPNAnalysis(mcn)

    K = mpna.omega2K(mpna.mcn.omega()[0])

    min_val = torch.min(K)
    max_val = torch.max(K)

    K_list = [K]

    for nc in range(num_checkpoints-1):
        
        mpna.train_mcn(timesteps=timesteps, lr=0.01)

        K = mpna.K_history[0][-1].to("cpu")

        K_list.append(K)

        min_val = min(min_val, torch.min(K))
        max_val = max(max_val, torch.max(K))

    for nc in range(num_checkpoints):
        im = ax[nc].imshow(K_list[nc], cmap='magma', vmin=min_val, vmax=max_val)
        ax[nc].axis('off')
        ax[nc].set_title('$t={}$'.format(nc))

    fig.tight_layout()
    fig.savefig('Figure4.pdf')

    plt.show()

