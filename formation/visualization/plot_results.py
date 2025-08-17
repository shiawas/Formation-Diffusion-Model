import matplotlib.pyplot as plt

def plot_results(t, x, v, ev, control):
    """
    Plot trajectories of x, v, ev, control vs time
    Arguments:
        t       : (N,) time array
        x       : (N,4) state trajectories
        v       : (N,4) velocity trajectories
        ev      : (N,4) auxiliary variable trajectories
        control : (N,4) control input trajectories
    """

    # Figure 1: x_i
    plt.figure(figsize=(8/2.54, 7/2.54))  # cm to inches
    plt.plot(t, x[:,0], 'g-', linewidth=1, label='$x_1$')
    plt.plot(t, x[:,1], 'b-', linewidth=1, label='$x_2$')
    plt.plot(t, x[:,2], 'r-', linewidth=1, label='$x_3$')
    plt.plot(t, x[:,3], 'k-', linewidth=1, label='$x_4$')
    plt.xlabel("Time [s]")
    plt.ylabel(r"$x_i, \; i \in \mathcal{V}$")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Figure 2: v_i
    plt.figure(figsize=(8/2.54, 7/2.54))
    plt.plot(t, v[:,0], 'g-', linewidth=1, label='$v_1$')
    plt.plot(t, v[:,1], 'b-', linewidth=1, label='$v_2$')
    plt.plot(t, v[:,2], 'r-', linewidth=1, label='$v_3$')
    plt.plot(t, v[:,3], 'k-', linewidth=1, label='$v_4$')
    plt.xlabel("Time [s]")
    plt.ylabel(r"$v_i, \; i \in \mathcal{V}$")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Figure 3: ev
    plt.figure(figsize=(8/2.54, 7/2.54))
    plt.plot(t, ev[:,0], 'g-', linewidth=1, label=r'$\mathbf{v}+\mathbf{\phi}_1$')
    plt.plot(t, ev[:,1], 'b-', linewidth=1)
    plt.plot(t, ev[:,2], 'r-', linewidth=1)
    plt.plot(t, ev[:,3], 'k-', linewidth=1)
    plt.xlabel("Time [s]")
    plt.ylabel(r"$\mathbf{v}+\mathbf{\phi}_1$")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Figure 4: control
    plt.figure(figsize=(8/2.54, 7/2.54))
    plt.plot(t, control[:,0], 'g-', linewidth=1, label='$u_1$')
    plt.plot(t, control[:,1], 'b-', linewidth=1, label='$u_2$')
    plt.plot(t, control[:,2], 'r-', linewidth=1, label='$u_3$')
    plt.plot(t, control[:,3], 'k-', linewidth=1, label='$u_4$')
    plt.xlabel("Time [s]")
    plt.ylabel(r"$u_i$")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.show()
