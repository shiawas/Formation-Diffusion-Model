import numpy as np
from formation.bearing_material.bearing_vector import br
from formation.bearing_material.signum_term import signum_term
from formation.bearing_material.projection import Pr
from formation.bearing_material.lambda_min import compute_lambda_min
from scipy.linalg import block_diag

def control_law_timevarying(p, v, H_bar, H, g_star, kp, kv, alpha):
    d , _ = p.shape
    m , _ = H.shape

    g = np.zeros((d, m))
    g_dot = np.zeros((d, m))

    for k in range(m):
        i = np.where(H[k, :] == -1)[0][0]
        j = np.where(H[k, :] ==  1)[0][0]
        pij = (p[:, j] - p[:, i]).reshape(d,1,order='F')
        vij = (v[:, j] - v[:, i]).reshape(d,1,order='F')
        norm_pij = np.linalg.norm(pij)
        if norm_pij > 1e-12:
            gk = pij / norm_pij
            g[:, k] = gk.flatten(order='F')
            Pg = Pr(gk)
            g_dot[:, k] = (Pg @ vij / norm_pij).flatten(order='F')


    e1 = (g - g_star).reshape(d*m,1,order='F')
    e2 = g_dot.reshape(d*m,1,order='F')

    term1 = -kp * (H_bar.T @ e1)
    term2 = -kv * (H_bar.T @ e2)
    term3 = -signum_term(H, p, g_star, alpha)

    u = term1 + term2 + term3
    return u




