import numpy as np
from formation.bearing_material.bearing_vector import br
from formation.bearing_material.signum_term import signum_term
from formation.bearing_material.projection import Pr
from formation.bearing_material.g_utils import compute_g_and_gdot
from scipy.interpolate import interp1d
from formation.data_transformation.load_leader_trajectories import load_leader_trajectories
from scipy.linalg import block_diag

# using np.gradient to calculate a1_traj from p1dot from .csv file

p1_func, p2_func, v1_func, v2_func, phi_func, phi_dot_func = load_leader_trajectories("combined_data.csv", dt=0.01)

def control_law_timevarying(t, x, H, g_star, kp, theta):

    d = 2 
    n = 4
    m , _ = H.shape

    p = x[:d*n]

    p = p.reshape(d, n,order='F')

    p[:, 0] = p1_func(t)
    p[:, 1] = p2_func(t)
    
    v = np.zeros_like(p)

    g, _ , _ = compute_g_and_gdot(H, p, v, eps=1e-9)

    g_flat = g.flatten(order='F')
    g_star_flat = g_star.flatten(order = 'F')

    P_blocks = [np.eye(d) - np.outer(g_star[:, k], g_star[:, k]) for k in range(m)]
    P_diag_star = block_diag(*P_blocks)
    
    H_bar = np.kron(H, np.eye(d))

    nl = 2
    nf = n - nl

    G = np.zeros((d*n, d*n))
    G[d*nl:, d*nl:] = np.eye(d*nf)

    q = P_diag_star @ g_flat
    s = np.sign(q)

    term1 = -kp * G @ H_bar.T @ (g_flat - g_star_flat)
    term2 = -theta * (G @ H_bar.T @ (P_diag_star @ s))

    u = term1 + term2 # (8,)
    
    # velocities of 2 leaders  
    v1 = v1_func(t)
    v2 = v2_func(t)


    dp = u.reshape(d, n, order='F')
    dp[:, 0] = v1
    dp[:, 1] = v2 

    return dp.flatten(order='F')




