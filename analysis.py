import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import os

from multichannel_net import MultipathwayNet

X_default = torch.eye(8)
Y_default = torch.Tensor([  [1,1,1,1,1,1,1,1],
                            [1,1,1,1,0,0,0,0],
                            [0,0,0,0,1,1,1,1],
                            
                            [1,1,0,0,0,0,0,0],
                            [0,0,1,1,0,0,0,0],
                            [0,0,0,0,1,1,0,0],
                            [0,0,0,0,0,0,1,1],
                            
                            [1,0,0,0,0,0,0,0],       
                            [0,1,0,0,0,0,0,0],
                            [0,0,1,0,0,0,0,0],
                            [0,0,0,1,0,0,0,0],
                            [0,0,0,0,1,0,0,0],
                            [0,0,0,0,0,1,0,0],
                            [0,0,0,0,0,0,1,0],
                            [0,0,0,0,0,0,0,1],
                            ]).T

Y_alt = torch.Tensor([  [0, 0, 0, 0, 0, 1, 1, 1],
                        [0, 1, 1, 1, 0, 1, 0, 1],
                        [0, 0, 1, 0, 1, 0, 1, 0],
                        [0, 1, 1, 1, 0, 1, 1, 0],
                        [0, 1, 0, 0, 1, 0, 1, 1],
                        [1, 1, 1, 0, 1, 0, 0, 1],
                        [0, 1, 1, 1, 0, 0, 0, 1],
                        [0, 0, 1, 1, 1, 1, 1, 0],
                        [1, 1, 1, 0, 0, 0, 1, 0],
                        [1, 1, 0, 0, 0, 1, 1, 1],
                        [1, 1, 1, 0, 0, 1, 0, 0],
                        [0, 1, 1, 1, 1, 1, 1, 0],
                        [0, 1, 1, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 1],
                        [0, 0, 1, 1, 1, 0, 0, 1]]).T

class MPNAnalysis (object):
    def __init__(self, mcn, X=X_default, Y=Y_default, device=None):

        self.mcn = mcn

        self.X = X
        self.Y = Y

        if device is None:
            if torch.has_mps:
                self.device = "mps"
            elif torch.has_cuda:
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print("Using device {}".format(self.device))

        # self.mcn.to(self.device)
        # self.X = self.X.to(self.device)
        # self.Y = self.Y.to(self.device)

        sigma_yx = self.Y.T.mm(self.X)/self.Y.shape[0]

        U,S,V = torch.svd(sigma_yx, some=False)

        self.U = U
        self.S = S
        self.V = V

        self.loss_history = None
        self.omega_history = None
        self.K_history = None

    def omega2K(self, omega):

        with torch.no_grad():
            k = omega.mm(self.V)
            k = self.U.T.mm(k)

        return k

    def train_mcn(self, timesteps=1000, lr=0.01):

        loss = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.SGD(params=self.mcn.parameters() , lr=lr)

        self.mcn.train()

        loss_history = []
        omega_history = []

        for t in range(timesteps):

            output = self.mcn(self.X)
            loss_val = loss(output, self.Y)

            loss_history.append(loss_val.to("cpu").detach().numpy())
            omega_history.append(self.mcn.omega())

            loss_val.backward()
            optimizer.step()

            optimizer.zero_grad()

        omega_history = zip(*omega_history)

        # convert omegas to Ks
        K_history = []
        for pathway in omega_history:
            K_history.append([self.omega2K(om) for om in pathway])


        self.loss_history = loss_history
        self.omega_history = omega_history
        self.K_history = K_history

        return loss_history, K_history

    def plot_K(self, ax, savedir='', labels=None, savename=None, savelabel='', min_val=0, max_val=2):

        if self.K_history is None:
            raise Exception("MultipathwayNet must be trained before visualization.")

        num_K = len(self.K_history)

        K_list = [pathway[-1].to("cpu") for pathway in self.K_history]

        # min_val = np.min([torch.min(K) for K in K_list])
        # max_val = np.max([torch.max(K) for K in K_list])
        
        if labels is None:
            labels = [i for i in range(len(K_list))]

        for i, K in enumerate(K_list):
            im = ax[i].imshow(K, vmin=min_val, vmax=max_val, cmap='magma')  # 'inferno'
            ax[i].set_title(r'$\bf K_{}$'.format(labels[i]), fontsize=20)
            ax[i].axis('off')

        plt.colorbar(im, ax=ax, shrink=0.6)

    def plot_K_history(self, ax, savename=None, D='unknown', savelabel=''):

        if self.K_history is None:
            raise Exception("MultipathwayNet must be trained before visualization.")

        num_pathways = len(self.K_history)
        timesteps = len(self.K_history[0])

        for i in range(min(self.mcn.input_dim, self.mcn.output_dim)):
        
            z1 = np.array([K[i,i].to("cpu") for K in self.K_history[0]])
            z2 = np.array([K[i,i].to("cpu") for K in self.K_history[1]])

            x = np.ones(timesteps)*i
            y = np.arange(timesteps)
            if i == 0:
                ax.plot3D(x, y, z1, 'C0', linewidth=4, label=r'$K_{a\alpha}$')
                line=ax.plot3D(x, y, z2, 'C1', linewidth=4, label=r'$K_{b\alpha}$')[0]
                line.set_dashes([1, 1, 1, 1])
            ax.plot3D(x, y, z1, 'C0', linewidth=4 )
            line=ax.plot3D(x, y, z2, 'C1', linewidth=4)[0]
            line.set_dashes([2, 1, 2, 1])  
        ax.tick_params(axis='x', labelsize= 10)
        ax.tick_params(axis='y', labelsize= 10)
        ax.tick_params(axis='z', labelsize= 10)
        ax.set_xlabel(r'dimension $\alpha$', fontsize=15)
        ax.set_ylabel('epoch', fontsize=15)
        ax.set_zlabel(r'$K_{a,b\alpha}$', fontsize=15)
        ax.legend(fontsize=17, loc='upper left')

        ax.set_box_aspect((2.25,1.75,1))


if __name__=='__main__':

    import argparse

    torch.manual_seed(345345)

    plt.rc('font', size=20)
    plt.rcParams['figure.constrained_layout.use'] = True
    import matplotlib
    matplotlib.rcParams["mathtext.fontset"] = 'cm'
    
    parser = argparse.ArgumentParser()

    # parser.add_argument('--timesteps', type=int, default=10000)
    # parser.add_argument('--nonlinearity', type=str, default='relu')

    args = parser.parse_args()

    nonlin = None
    # if args.nonlinearity=='relu':
    #     nonlin = torch.nn.ReLU()
    # if args.nonlinearity=='tanh':
    #     nonlin = torch.nn.Tanh()

    # timesteps = args.timesteps

    depth_list = [2,3,4,7]

    fig_train, ax_train = plt.subplots(1,len(depth_list), figsize=(25,8))
    ax_train[0].set_ylabel('training error')
    
    fig_history = plt.figure(figsize=(24,10))
    gs = gridspec.GridSpec(2, 6,width_ratios=[2.2,1,1,2.2,1,1],figure=fig_history)

    timestep_list = [1000, 1000, 1400, 10000]

    min_val = 0.0
    max_val = 1.0

    for di, depth in enumerate(depth_list):

        ax3d = fig_history.add_subplot(gs[di*3], projection='3d')
        ax2 = fig_history.add_subplot(gs[di*3 +1])
        ax3 = fig_history.add_subplot(gs[di*3 +2])

        mcn = MultipathwayNet(8,15, depth=depth, num_pathways=2, width=1000, bias=False, nonlinearity=nonlin)
        mpna = MPNAnalysis(mcn, Y=Y_default)
        mpna.train_mcn(timesteps=timestep_list[di], lr=0.01)

        ax_train[di].plot(mpna.loss_history)
        ax_train[di].set_xlabel('epoch')
        ax_train[di].set_title("$D={}$".format(depth))

        ax3d.set_title("$D=2$")
        mpna.plot_K_history(ax3d, D=depth)

        K_list = [pathway[-1].to("cpu") for pathway in mpna.K_history]
        min_val_temp = np.min([torch.min(K) for K in K_list])
        max_val_temp = np.max([torch.max(K) for K in K_list])

        min_val = min(min_val_temp, min_val)
        max_val = max(max_val_temp, max_val)


    for di, depth in enumerate(depth_list):
        mpna.plot_K([ax2,ax3], labels=['a', 'b'], min_val=min_val, max_val=max_val)

    fig_train.suptitle("Training loss")
    fig_train.savefig('train_loss.pdf')
    fig_history.savefig('test.pdf')

    plt.show()

