import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng

plt.rc('font', size=20)

from scipy.interpolate import interp1d

rng = default_rng(seed=49827345)

def K_full(p,D):
    return p*np.power(p**2+1, (D-1)/2)

def K_asym(p,D):
    return p**D

p = np.linspace(0,10, 10000)
K = K_full(p,5)

pOfK = interp1d(K, p)

def get_init_k(n, s=0.1):
    return rng.normal(0,s,size=(n,2))

def k_trajectory(D, S, p0):

    px0, py0 = p0
    dt = 0.001

    T = 10000

    px = np.zeros(T)
    py = np.zeros(T)

    px[0] = px0
    py[0] = py0

    Kx = np.zeros(T)
    Ky = np.zeros(T)

    Kx[0] = K_full(px0, D)
    Ky[0] = K_full(py0, D)

    for t in range(1,T):

        Omega = S - Kx[t-1] - Ky[t-1]
        px[t] = px[t-1] + dt*(np.sqrt(px[t-1]**2+1)**(D-1))*Omega
        py[t] = py[t-1] + dt*(np.sqrt(py[t-1]**2+1)**(D-1))*Omega

        Kx[t] = K_full(px[t], D)
        Ky[t] = K_full(py[t], D)

    return Kx, Ky

kx = np.arange(0.01,10.01, 0.5)
ky = np.arange(0.01,10.01, 0.5)

KX,KY = np.meshgrid(kx, ky)

PX = pOfK(KX)
PY = pOfK(KY)

def k_vector_field(D, S):
    Omega = S - KX - KY

    dkx = ( (KX/PX)**2 + (D-1)*PX*((KX/PX)**((D-3)/(D-1)))*KX  )*Omega
    dky = ( (KY/PY)**2 + (D-1)*PY*((KY/PY)**((D-3)/(D-1)))*KY  )*Omega

    return dkx, dky

S = 10.

std_list = [0.1, 0.01]
fig, ax = plt.subplots(2,4, figsize=(26,13))

for si, std in enumerate(std_list):
    

    D_vals = [2,3,8,13]

    p_init_array = get_init_k(100, s=std)

    for i, D in enumerate(D_vals):

        ax[si,i].set_ylim([0,10])
        ax[si,i].set_xlim([0,10])

        for p_init in p_init_array:
            k_traj = k_trajectory(D, S, p_init)
            ax[si,i].plot(k_traj[0], k_traj[1],'k')

        dkx, dky = k_vector_field(D, S)

        ax[si,i].plot(kx, S-kx, 'b-')
        ax[si,i].streamplot(KX,KY, dkx, dky, density=1.0, linewidth=None, color='#A23BEC')
        ax[si,i].set_ylabel(r'$K_{2\alpha}$')
        ax[si,i].set_xlabel(r'$K_{1\alpha}$')
        ax[0,i].set_title('$D={}$'.format(D))

fig.tight_layout()
fig.savefig("Figure3.pdf")

plt.show()
