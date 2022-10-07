import numpy as np
import matplotlib.pyplot as plt
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
    def __init__(self, mcn, X=X_default, Y=Y_default):

        self.mcn = mcn

        self.X = X
        self.Y = Y

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

        # omega = self.mcn.omega()

        # with torch.no_grad():
        #     omega_hat = []
        #     for om in omega:
        #         om_hat = om.mm(self.V.T)
        #         om_hat = self.U.T.mm(om_hat)

        #         omega_hat.append(om_hat)

        # return omega_hat


    def train_mcn(self, timesteps=1000, lr=0.01):

        loss = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.SGD(params=self.mcn.parameters() , lr=lr)

        self.mcn.train()

        loss_history = []
        omega_history = []

        for t in range(timesteps):

            output = self.mcn(self.X)
            loss_val = loss(output, self.Y)

            loss_history.append(loss_val.detach().numpy())
            omega_history.append(mcn.omega())

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

    def plot_K(self, savedir='', labels=None, savename=None, savelabel=''):

        if self.K_history is None:
            raise Exception("MultipathwayNet must be trained before visualization.")

        num_K = len(self.K_history)

        fig, ax = plt.subplots(1,num_K, figsize=(5,5))

        K_list = [pathway[-1] for pathway in self.K_history]

        min_val = np.min([torch.min(K) for K in K_list])
        max_val = np.max([torch.max(K) for K in K_list])
        
        if labels is None:
            labels = [i for i in range(len(K_list))]

        for i, K in enumerate(K_list):
            im = ax[i].imshow(K, vmin=min_val, vmax=max_val, cmap='bwr')
            ax[i].set_title(r'$\bf K_{}$'.format(labels[i]), fontsize=20)
            ax[i].axis('off')

        cb_ax = fig.add_axes([0.94,.2,.04,.6])
        cbar=fig.colorbar(im,orientation='vertical',cax=cb_ax)
        cbar.ax.tick_params(labelsize=15) 

        if savename is None:
            savename = 'final_K'
        savename = savename + '.pdf'
        if len(savelabel)>0:
            savename = savelabel+'_'+savename
        savepath = os.path.join(savedir, savename)
        
        fig.savefig(savepath, bbox_inches='tight')

    def plot_K_history(self, savename=None, D='unknown', savelabel=''):

        if self.K_history is None:
            raise Exception("MultipathwayNet must be trained before visualization.")

        num_pathways = len(self.K_history)
        timesteps = len(self.K_history[0])

        fig = plt.figure(figsize=(8,5))
        ax = plt.axes(projection='3d')
        for i in range(min(self.mcn.input_dim, self.mcn.output_dim)):
        
            z1 = np.array([K[i,i] for K in self.K_history[0]])
            z2 = np.array([K[i,i] for K in self.K_history[1]])

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
        plt.legend(fontsize=17, loc='upper left')
        # plt.title(r'$N_w=%d$'%num_pathways, fontsize=20)
        fig.suptitle('$D={}$'.format(D))
        if len(savelabel)>0:
            savename = savelabel+'_history.pdf'
        else:
            savename = 'history.pdf'
        fig.savefig(savename)


if __name__=='__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--depth', type=int)
    parser.add_argument('--timesteps', type=int, default=10000)
    parser.add_argument('--nonlinearity', type=str, default='relu')

    args = parser.parse_args()

    nonlin = None
    if args.nonlinearity=='relu':
        nonlin = torch.nn.ReLU()
    if args.nonlinearity=='tanh':
        nonlin = torch.nn.Tanh()

    depth = args.depth
    timesteps = args.timesteps

    savelabel = '{}_{}'.format(args.nonlinearity, depth)

    mcn = MultipathwayNet(8,15, depth=depth, num_pathways=2, width=1000, bias=False, nonlinearity=nonlin)

    mpna = MPNAnalysis(mcn, Y=Y_default)

    mpna.train_mcn(timesteps=timesteps, lr=0.01)

    fig, ax = plt.subplots()
    ax.plot(mpna.loss_history)
    fig.savefig(savelabel+'_train_loss.pdf')

    if len(mpna.K_history)>1:
        mpna.plot_K(labels=['a', 'b'], savelabel=savelabel)

    if len(mpna.K_history)==2:
        mpna.plot_K_history(D=depth, savelabel=savelabel)

    plt.show()

