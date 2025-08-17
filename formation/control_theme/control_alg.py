import numpy as np
from formation.bearing_material.heading_unit import h_i
from formation.bearing_material.heading_pre import hp_i
from formation.bearing_material.dheading import dh_i
from formation.bearing_material.bearing_vector import br
from formation.bearing_material.signum_term import signum_term
from scipy.linalg import block_diag

def update_law_robot(t, x, n, d, m, H, H_bar, Z, k1, k2, beta_c, g_star, v0):
    p = x[:d*n]
    xi = x[d*n:d*n+d*n]
    theta = x[2*d*n:]

    # reshape
    p_mat = p.reshape(d, n,order='F')
    xi_mat = xi.reshape(d, n,order='F')

    # Dlkblock
    blocks_h = [np.outer(h_i(theta[i]), h_i(theta[i])) if i < n else np.zeros((d,d)) for i in range(n)]
    D_h = block_diag(*blocks_h)

    blocks_hp = [np.outer(hp_i(theta[i]), hp_i(theta[i])) if i < n else np.zeros((d,d)) for i in range(n)]
    D_hp = block_diag(*blocks_hp)

    z = (H_bar @ p).reshape(d,m,order='F')
    g = np.concatenate([br(z[:, i]) for i in range(m)])

    del_g = g - g_star.flatten(order='F')

    # control input
    sig_u = signum_term (H, p_mat, g_star, beta_c)
    dp = np.concatenate([
        (v0[0]*h_i(theta[0])).reshape(-1,order='F'),
        (v0[1]*h_i(theta[1])).reshape(-1,order='F'),
        np.zeros(d * (n-2))
    ]) - Z @ D_h @ (k1 * (H_bar.T @ del_g) - xi.flatten(order='F')) - sig_u.flatten(order='F')

    dxi = -(Z @ D_h @ H_bar.T @ del_g) - Z @ D_hp @ xi.flatten(order='F')

    r = ((H_bar.T @ del_g) - xi.flatten(order='F')).reshape(d, n,order='F')

    omega = -k2 * np.array([
        0.0, 0.0,
        (hp_i(theta[2]).T @ r[:,2]).item(),
        (hp_i(theta[3]).T @ r[:,3]).item()
    ])
    dp = dp.flatten(order='F')
    dxi = dxi.flatten(order='F')
    omega = omega.flatten(order='F')

    return np.concatenate([dp,dxi,omega])