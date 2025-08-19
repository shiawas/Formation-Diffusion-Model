import numpy as np

def make_g_list_and_norm(H, p): 
    m , _ = H.shape
    d = 2
    g_list = []
    norm_list = []
    eps_norm = 1e-12

    for k in range(m):
        i = np.where(H[k, :] == -1)[0][0]
        j = np.where(H[k, :] ==  1)[0][0]
        vec_ij = ( p[:, j] - p[:, i] )
        nrm = np.linalg.norm(vec_ij)
        norm_list.append(nrm)
        if nrm > eps_norm:
            g_list.append(vec_ij / nrm)
        else : 
            g_list.append(np.zeros(d))

    return g_list, norm_list