import numpy as np
from scipy.integrate import solve_ivp
from formation.bearing_material.heading_unit import h_i
from formation.bearing_material.heading_pre import hp_i
from formation.bearing_material.dheading import dh_i
from formation.bearing_material.bearing_vector import br#intial paramaters
from formation.bearing_material.signum_term import signum_term

n = 4
d = 2
m = 5
T = 300
dt = 0.1

# positions
p0 = np.array([
    [10.0000, 10.0000, 4.9382, 4.5363],
    [0, 5.0000, 4.4845, 0.2687]
], dtype=float)

# leader's velocity
uc = 0.15
v0 = np.array([uc, uc, 0, 0], dtype=float)
v0 = v0.reshape(-1, 1)


# headings
theta0 = np.array([np.pi/6, np.pi/6, 0, np.pi/3], dtype=float)
theta0 = theta0.reshape(-1, 1)


# angular velocities
omega0 = np.zeros([n,1])


# integral terms
xi0 = np.zeros((d, n))

# desired bearing vectors (g_star)
g_star = np.array([
    [0, -1/np.sqrt(2), -1, -1, 0],
    [1,  1/np.sqrt(2), 0,  0, -1]
])

# incidence matrix
H = np.array([
    [-1, 1, 0, 0],
    [-1, 0, 1, 0],
    [0, -1, 1, 0],
    [-1, 0, 0, 1],
    [0, 0, -1, 1]
])

L = H.T @ H
H_bar = np.kron(H, np.eye(d))
L = np.kron(L, np.eye(d))
Z = np.block([
    [np.zeros((2*d, 2*d)), np.zeros((2*d, d*(n-2)))],
    [np.zeros((d*(n-2), 2*d)), np.eye(d*(n-2))]
])

#control gains
k1 = 15
k2 = 7
beta_c = 0.15

sig_u = signum_term(H, p0, g_star, beta_c)
print(sig_u)