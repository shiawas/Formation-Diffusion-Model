import numpy as np
from formation.bearing_material.projection import Pr

def make_dot_g_list(H, v, g_list, norm_list):
    m, _ = H.shape
    d = v.shape[0]
    dot_g_list = []
    idx = 0
    for k in range(m):
        i_idx = np.where(H[k, :] == -1)[0]
        j_idx = np.where(H[k, :] == 1)[0]
        if len(i_idx) == 0 or len(j_idx) == 0:
            continue
        i = i_idx[0]
        j = j_idx[0]
        dot_z = v[:, j] - v[:, i]
        g_k = g_list[idx]
        P_g = Pr(g_k)
        nrm = norm_list[idx]
        if nrm > 0:
            dot_g = P_g @ dot_z / nrm
        else:
            dot_g = np.zeros(d)
        dot_g_list.append(dot_g)
        idx += 1
    return dot_g_list