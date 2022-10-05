import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng

from scipy.interpolate import interp1d

rng = default_rng()

def get_init_k(n):
    return rng.uniform(0,0.01,size=(n,2))

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

fig, ax = plt.subplots(1,4, figsize=(40,6))

D_vals = [3,9,15,21]

p_init_array = get_init_k(100)

for i, D in enumerate(D_vals):

    ax[i].set_ylim([0,10])
    ax[i].set_xlim([0,10])

    for p_init in p_init_array:
        k_traj = k_trajectory(D, S, p_init)
        ax[i].plot(k_traj[0], k_traj[1],'k')

    dkx, dky = k_vector_field(D, S)

    ax[i].plot(kx, S-kx, 'k')
    ax[i].streamplot(KX,KY, dkx, dky, density=1.0, linewidth=None, color='#A23BEC')
    # ax[i].quiver(KX,KY, dkx, dky, linewidth=None, color='#A23BEC')
    ax[i].set_ylabel('K_2')
    ax[i].set_xlabel('K_1')
    ax[i].set_title('D={}'.format(D))

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