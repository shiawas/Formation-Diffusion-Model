import numpy as np
from scipy.linalg import block_diag

def compute_alpha_bound(H, g_list, nf, xi_a):
    """
    Compute lower bound on alpha according to formula.
    
    Parameters
    ----------
    H : (m, n) incidence matrix
    g_list : list of (d,) bearing unit vectors
    nf : number of followers
    xi_a : scalar parameter
    
    Returns
    -------
    alpha_min : float
    """
    d = g_list[0].shape[0]
    m, n = H.shape
    
    M = H[:, -nf:]
    M_bar = np.kron(M, np.eye(d))   # (dm, d*nf)

    # Build diag(P_gk)
    P_blocks = [np.eye(d) - np.outer(g, g) for g in g_list]
    P_diag = block_diag(*P_blocks)   # (dm, dm)

    a = M_bar.T @ P_diag
    norm_val = np.linalg.norm(a, 2)   # spectral norm

    B = M_bar.T @ P_diag @ P_diag @ M_bar
    eigvals = np.linalg.eigvalsh(B)   
    lam_min = np.min(eigvals[eigvals > 1e-9])  # smallest positive eigenvalue

    # alpha bound
    alpha_min = np.sqrt(nf) * xi_a * norm_val / lam_min
    return alpha_min    