import numpy as np
from formation.bearing_material.bearing_vector import br
from formation.bearing_material.signum_term import signum_term
from formation.bearing_material.projection import Pr
from formation.bearing_material.g_utils import compute_g_and_gdot

from scipy.linalg import block_diag

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
    
    vr = np.array([
        [0.35, 0.35], 
        [0.3*np.cos(0.03*t), 0.3*np.cos(0.03*t)]
    ])

    vr_dot = np.array([
        [0.0, 0.0],
        [-0.009*np.sin(0.03*t), -0.009*np.sin(0.03*t)]
    ])

    dp = v
    dv = u.reshape(d, n, order='F')
    dp[:, :nl] = vr
    dp[:, :nl] = vr_dot

    dp = dp.flatten(order='F')
    dv = dv.flatten(order='F')

    return np.concatenate([dp,dv])




