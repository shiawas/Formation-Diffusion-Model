#conda run --live-stream --name maxs python -m formation.control_theme.main_robot


import numpy as np
from scipy.integrate import solve_ivp
from formation.bearing_material.heading_unit import h_i
from formation.bearing_material.heading_pre import hp_i
from formation.bearing_material.dheading import dh_i
from formation.bearing_material.bearing_vector import br
from formation.control_theme.control_alg import update_law_robot
from formation.visualization.plot_robot_traj import plot_robot_traj

#intial paramaters
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
beta_c = 0.2

#intial state vector
x0 = np.concatenate([p0.flatten(order='F'), xi0.flatten(order='F'), theta0.flatten(order='F')])

#ODE
def ode_system(t, x):
    return update_law_robot(t, x, n, d, m, H, H_bar, Z, k1, k2, beta_c, g_star, v0)


sol = solve_ivp(ode_system, [0, T], x0, method = 'RK45', max_step = dt)
t = sol.t
p = sol.y[:d*n, :] #shape (8,3006)
xi = sol.y[d*n:d*n+d*n, :] #shape (8,3006)
theta = sol.y[2*d*n:, :] #shape (4,3006)

p = p.T
xi = xi.T
theta = theta.T
N = p.shape[0]
e_g = []
v = np.zeros((N, n))

#approved
for i in range(N):
    z = np.reshape(H_bar @ p[i,:].T, (d, m), order='F') # v
    g = np.concatenate([br(z[:, j]) for j in range(5)], axis=0) # v
    e_g.append(np.linalg.norm(g - g_star.reshape(-1,order='F'))) # v
    r = -(k1 * (H_bar.T @ (g - g_star.reshape(-1,order='F'))) - xi[i, :].T) # v
    r = np.reshape(r, (d, n), order='F') # v
    hc = h_i(theta[0,0]) * uc
    v[i, :] = np.array([
        uc,
        uc,
        np.linalg.norm((h_i(theta[0, 2]) @ h_i(theta[0, 2]).T @ r[:, 2]).reshape(-1,1) - hc),
        np.linalg.norm((h_i(theta[0, 3]) @ h_i(theta[0, 3]).T @ r[:, 3]).reshape(-1,1) - hc)
    ])

'''

'''
plot_robot_traj(t, p, theta, v, e_g, n, d, T, scale=8, save_video=True)
