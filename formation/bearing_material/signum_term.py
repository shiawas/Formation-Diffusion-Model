import numpy as np
from formation.bearing_material.lambda_min import compute_lambda_min
from formation.bearing_material.g_utils import compute_g_and_gdot
from formation.bearing_material.projection import Pr


def signum_term(H, p, g_star, beta_c):
    """
    H: (m x n) numpy array (incidence matrix)
    p: (2 x n) numpy array (positions of nodes)
    g_star: (2 x m) numpy array (desired bearing vectors)
    beta_c: scalar
    return: sig_u (2*n, 1) numpy array
    """
    m, n = H.shape
    d = 2
    sig_u = np.zeros((d*n, 1))

    leaders = [0, 1]  # Python index (MATLAB 1,2)
    followers = [i for i in range(n) if i not in leaders]

    g_list, _ , _ = compute_g_and_gdot(H, p)

    for i in followers:
        lambda_i = compute_lambda_min(i, H, g_list)
        k_i2 = lambda_i ** (-0.5) if lambda_i > 0 else 0.0

        sum_vec = np.zeros((d,1))

        for k in range(m):
            if H[k, i] == -1:
                gij = g_list[k]
                Pgij = Pr(gij)
                gstar_ij = g_star[:, k].reshape(d,1,order = 'F')

                sum_vec += Pgij @ np.sign(Pgij @ gstar_ij)

        sig_u[i*d:(i+1)*d, :] = k_i2 * beta_c * sum_vec

    return sig_u