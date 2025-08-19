import numpy as np

def compute_g_and_gdot(H, p, v, eps=1e-9):
    d, n = p.shape
    m = H.shape[0]
    I_d = np.eye(d)

    H_bar = np.kron(H, I_d)

    p_vec = p.reshape(-1, order='F')
    v_vec = v.reshape(-1, order='F')

    z = (H_bar @ p_vec).reshape(m, d).T
    z_dot = (H_bar @ v_vec).reshape(m, d).T

    norms = np.linalg.norm(z, axis=0, keepdims=True)
    norms = np.maximum(norms, eps)
    g = z / norms

    proj = z_dot - g * np.sum(g * z_dot, axis=0, keepdims=True)
    g_dot = proj / norms

    return g, g_dot, norms
