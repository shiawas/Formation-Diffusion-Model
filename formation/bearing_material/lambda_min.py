import numpy as np
from formation.bearing_material.projection import Pr

def compute_lambda_min(i, H, g_list):
    m, _ = H.shape
    Mi = np.zeros((2,2))

    for k in range(m):
        if H[k, i] != 0:
            gij = g_list[k]
            if np.any(gij): 
                Pgij = Pr(gij)
                Mi += Pgij

    # Tính lambda_1 sau khi cộng hết tất cả các Pgij
    if np.all(Mi == 0):
        lambda_1 = 0.0
    else:
        eig_vals = np.linalg.eigvals(Mi)
        lambda_1 = np.min(eig_vals.real)
        if lambda_1 < 0:
            lambda_1 = max(lambda_1, 0.0)

    return lambda_1
