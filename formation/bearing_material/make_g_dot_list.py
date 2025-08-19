import numpy as np
from formation.bearing_material.projection import Pr

def make_dot_g_list(H, v, g_list, norm_list):
    m, _ = H.shape
    d = v.shape[0]
    dot_g_list = []
    for k in range(m):
        i = np.where(H[k, :] == -1)[0][0]
        j = np.where(H[k, :] == 1)[0][0]
        dot_z = v[:, j] - v[:, i]  
        g_k = g_list[k]            
        P_g = Pr(g_k)             
        nrm = norm_list[k]
        if nrm > 0:
            dot_g = P_g @ dot_z / nrm 
        else:
            dot_g = np.zeros(d)      
        dot_g_list.append(dot_g)  
    return dot_g_list