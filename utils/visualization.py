import numpy as np
from matplotlib import pyplot as plt


def visualize_quadrotor_simulation_result(
    quadrotor, states: np.ndarray, obs_center: np.ndarray, obs_radius: np.ndarray
):
    def plot_obstacles(obs_center, obs_radius):
        for obs_p, obs_r in zip(obs_center, obs_radius):
            circle = plt.Circle(
                tuple(obs_p),
                obs_r,
                color="grey",
                fill=True,
                linestyle="--",
                linewidth=2,
                alpha=0.5,
            )
            plt.gca().add_artist(circle)

    plt.figure()
    plt.gca().set_aspect("equal", adjustable="box")


    # unpack state
    xs = [s[0] for s in states]  
    ys = [s[2] for s in states]
    phis = [s[4] for s in states]

    # obstacles
    plot_obstacles(obs_center, obs_radius)

    # compute trajectory at FRONT (not center)
    front_xs = [x + quadrotor.l_q * np.cos(phi) for x, phi in zip(xs, phis)]
    front_ys = [y + quadrotor.l_q * np.sin(phi) for y, phi in zip(ys, phis)]

    # plot trajectory using FRONT
    plt.scatter(front_xs, front_ys, s=10, color="blue", alpha=0.6)

    # plot quadrotor poses
    for (front_x, front_y, phi) in zip(front_xs[::10], front_ys[::10], phis[::10]):

        hx = np.cos(phi)
        hy = np.sin(phi)

        px = -np.sin(phi)
        py = np.cos(phi)

        # half width vector (perpendicular to heading)
        dx = (quadrotor.w_q / 2) * px
        dy = (quadrotor.w_q / 2) * py


        # 4 corners of the rectangle
        p1 = (front_x + dx, front_y + dy)  # front-left # 
        p2 = (front_x - dx, front_y - dy)  # front-right # 
        p3 = (p2[0] - 2*quadrotor.l_q*hx, p2[1] - 2*quadrotor.l_q*hy)  # back-right #
        p4 = (p1[0] - 2*quadrotor.l_q*hx, p1[1] - 2*quadrotor.l_q*hy)  # back-left # 

        rect_x = [p1[0], p2[0], p3[0], p4[0], p1[0]]
        rect_y = [p1[1], p2[1], p3[1], p4[1], p1[1]]
        plt.plot(rect_x, rect_y, color="red", alpha=0.6)

    # plot start and target using FRONT
    init_x, init_y = front_xs[0], front_ys[0]
    plt.scatter(init_x, init_y, s=200, color="green", alpha=0.75, label="init. position")
    plt.scatter(5.0, 5.0, s=200, color="purple", alpha=0.75, label="target position")

    plt.xlim(-1, 7)
    plt.ylim(-1, 7)
    plt.grid()
    plt.show()