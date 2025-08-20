import numpy as np
from scipy.linalg import block_diag

def compute_theta_bound(H, g_star, nf, xi_v):
    '''
    array to glist
    g_list = [g_star[:, i] for i in range(g_star.shape[1])]
    '''
    """
    Compute lower bound on alpha according to formula.
    
    Parameters
    ----------
    H : (m, n) incidence matrix
    g_list : list of (d,) bearing unit vectors dtype :list
    nf : number of followers
    xi_a : scalar parameter
    
    Returns
    -------
    alpha_min : float
    """
    g_list = [g_star[:, i] for i in range(g_star.shape[1])]
    d = g_list[0].shape[0]
    m, n = H.shape
    
    M = H[:, -nf:]
    M_bar = np.kron(M, np.eye(d))   # (dm, d*nf)

    # Build diag(P_gk)
    P_blocks = [np.eye(d) - np.outer(g, g) for g in g_list]
    P_diag = block_diag(*P_blocks)   # (dm, dm)

    B = M_bar.T @ P_diag @ P_diag @ M_bar
    eigvals = np.linalg.eigvalsh(B)   
    lam_min = np.min(eigvals[eigvals > 1e-9])  # smallest positive eigenvalue
    lam_max = np.max(eigvals[eigvals > 1e-9])

    # alpha bound
    theta_min = np.sqrt(nf * lam_max) * xi_v / lam_min
    return theta_min    