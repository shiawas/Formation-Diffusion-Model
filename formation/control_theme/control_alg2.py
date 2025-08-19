import numpy as np
from formation.bearing_material.bearing_vector import br
from formation.bearing_material.signum_term import signum_term
from formation.bearing_material.projection import Pr
from formation.bearing_material.g_utils import compute_g_and_gdot
from scipy.interpolate import interp1d
from formation.data_transformation import load_leader_trajectories

# using np.gradient to calculate a1_traj from p1dot from .csv file

v1_func, v2_func, a1_func, a2_func = load_leader_trajectories("combined_data.csv", dt=0.1)

def control_law_timevarying(t, x, H, g_star, kp, kv, alpha):

    d = 2 
    n = 4
    m , _ = H.shape

    p = x[:d*n]
    v = x[d*n : ]

    p = p.reshape(d, n,order='F')
    v = v.reshape(d, n,order='F')

    g, g_dot, norms = compute_g_and_gdot(H, p, v, eps=1e-9)

    g_flat = np.concatenate(g)
    dot_g_flat = np.concatenate(g_dot)
    
    g_star_flat = g_star.flatten(order = 'F')
    sign_dot_g = np.sign(dot_g_flat)
    H_bar = np.kron(H, np.eye(d))

    nl = 2
    nf = n - nl

    G = np.zeros((d*n, d*n))
    G[d*nl:, d*nl:] = np.eye(d*nf)

    term1 = -kp *   G @ H_bar.T @ (g_flat - g_star_flat)
    term2 = -kv * G @ H_bar.T @ dot_g_flat
    term3 = -alpha * G @ H_bar.T @ sign_dot_g

    u = term1 + term2 + term3 # (8,)
    
    v1 = v1_func(t)
    v2 = v2_func(t)
    a1 = a1_func(t)
    a2 = a2_func(t)

    vr = np.column_stack([v1, v2])
    vr_dot = np.column_stack([a1, a2])

    dp = v
    dv = u.reshape(d, n, order='F')
    dp[:, :nl] = vr
    dp[:, :nl] = vr_dot

    dp = dp.flatten(order='F')
    dv = dv.flatten(order='F')

    return np.concatenate([dp,dv])




