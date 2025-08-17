import numpy as np

def make_g_list(H, p): 
    m , _ = H.shape
    d = 2
    g_list = []
    eps_norm = 1e-12

    for k in range(m):
        i = np.where(H[k, :] == -1)[0][0]
        j = np.where(H[k, :] ==  1)[0][0]
        vec_ij = ( p[:, j] - p[:, i] ).reshape(d, 1, order = 'F')
        nrm = np.linalg.norm(vec_ij)
        if nrm > eps_norm:
            g_list.append(vec_ij / nrm)
        else : 
            g_list.append(np.zeros(d))

    return g_list