import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng

plt.rc('font', size=20)

from scipy.interpolate import interp1d

rng = default_rng()

# fig, ax = plt.subplots()
def K_full(p,D):
    return p*np.power(p**2+1, (D-1)/2)

def K_asym(p,D):
    return p**D

p = np.linspace(0,10, 10000)
K = K_full(p,5)

pOfK = interp1d(K, p)

# check the interpolation
# fig, ax = plt.subplots()
# plt.plot(K, p)
# k_vals = np.linspace(0,10,100)
# plt.plot(k_vals, pOfK(k_vals))

# plt.show()

# kx = np.arange(0.01,10.01,0.01)
# ky = np.arange(0.01,10.01,0.01)

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

fig, ax = plt.subplots(1,4, figsize=(40,6.5))

D_vals = [2,3,8,13]

p_init_array = get_init_k(100)

for i, D in enumerate(D_vals):

    ax[i].set_ylim([0,10])
    ax[i].set_xlim([0,10])

    for p_init in p_init_array:
        k_traj = k_trajectory(D, S, p_init)
        ax[i].plot(k_traj[0], k_traj[1],'k')

    dkx, dky = k_vector_field(D, S)

    ax[i].plot(kx, S-kx, 'b-')
    ax[i].streamplot(KX,KY, dkx, dky, density=1.0, linewidth=None, color='#A23BEC')
    # ax[i].quiver(KX,KY, dkx, dky, linewidth=None, color='#A23BEC')
    ax[i].set_ylabel(r'$K_{2\alpha}$')
    ax[i].set_xlabel(r'$K_{1\alpha}$')
    ax[i].set_title('$D={}$'.format(D))

plt.show()


# # ax.plot(p, (K_full(p, 5)- K_asym(p,5))/K_full(p, 5))
# # # ax.plot(p, K_asym(p, 5))

# p_x = np.arange(0,2,0.1)
# p_y = p_x

# X, Y = np.meshgrid(p_x, p_y)



# def p_vector_field(D, S):
#     Omega = S - K_full(X, D) - K_full(Y, D)



#     dx = np.power(X**2 + 1, D-1)*Omega
#     dy = np.power(Y**2 + 1, D-1)*Omega

#     # print(dx.shape, dy.shape)

#     return dx, dy

# def k_vector_field(D, S):
#     pass

# fig, ax = plt.subplots()

# dp_x, dp_y = p_vector_field(5, 10)

# ax.streamplot(X,Y,dp_x,dp_y, density=1.4, linewidth=None, color='#A23BEC')

# plt.show()


# S = 10
# D = 5


# x = np.arange(100)*S/100
# y = np.arange(100)*S/100

# X, Y = np.meshgrid(x,y)


# # print(Omega.shape)

# # print((X**c).shape)

# def asymptotic_vector_field(D, S=S):
#     c = 2. - 2./D
#     Omega = S - X - Y
#     K_x = D*(X**c)*Omega
#     K_y = D*(Y**c)*Omega

#     return K_x, K_y

# # print(X.shape, Y.shape)
# # print(K_x.shape, K_y.shape)

# fig, ax = plt.subplots()
# ax.plot(x, S-x)
# K_x, K_y = asymptotic_vector_field(D)
# ax.streamplot(X,Y,K_x,K_y, density=1.4, linewidth=None, color='#A23BEC')

# plt.show()