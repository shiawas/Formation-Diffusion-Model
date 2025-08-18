import numpy as np

def states_to_p1p2(states: np.ndarray, l_q: float, w_q: float):
    xs   = states[:, 0]
    xs_dot = states[:, 1]
    ys   = states[:, 2]
    ys_dot = states[:, 3]
    phis = states[:, 4]
    phis_dot = states[:, 5]
    # front points
    front_xs = xs + l_q * np.cos(phis)
    front_ys = ys + l_q * np.sin(phis)

    front_xs_dot = xs_dot - l_q * phis_dot * np.sin(phis)
    front_ys_dot = ys_dot + l_q * phis_dot * np.cos(phis)

    # half-width offsets
    dx = (w_q / 2) * -np.sin(phis)
    dy = (w_q / 2) *  np.cos(phis)

    dx_dot = (w_q / 2) * phis_dot * -np.cos(phis)
    dy_dot = (w_q / 2) * phis_dot * np.sin(phis) 

    p1 = np.stack([front_xs + dx, front_ys + dy], axis=1)
    p2 = np.stack([front_xs - dx, front_ys - dy], axis=1)

    p1_dot = np.stack([front_xs_dot + dx_dot, front_ys_dot + dy_dot], axis=1)
    p2_dot = np.stack([front_xs_dot - dx_dot, front_ys_dot - dy_dot], axis=1)

    return p1, p1_dot, p2, p2_dot, phis, phis_dot


