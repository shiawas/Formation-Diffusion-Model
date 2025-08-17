import numpy as np
import matplotlib.pyplot as plt
from formation.visualization.DrawUnicycle import draw_unicycle
import matplotlib.animation as animation

def plot_robot_traj(t, p, theta, v, e_g, n, d, T, scale=8, save_video=False):
    """
    Plot robot trajectories and formations (Python version of plot_robot_traj.m)

    Args:
        t       : (N,) time array
        p       : (N, d*n) positions of agents
        theta   : (N, n) heading angles
        v       : (N, n) velocities
        e_g     : (N,) formation error norm
        n       : number of agents
        d       : dimension (2)
        T       : simulation time
        scale   : scale factor for DrawUnicycle
        save_video : whether to save a video (default False)
    """
    N = p.shape[0]

    # --- 1. Trajectories + formations ---
    plt.figure()
    plt.title("Robot trajectories & formations")
    plt.plot(p[:,0], p[:,1], '-', color=[0.3,0.3,0.3], linewidth=0.5)
    plt.plot(p[:,2], p[:,3], '-', color=[0.3,0.3,0.3], linewidth=0.5)
    plt.plot(p[:,4], p[:,5], '-', color=[0.3,0.3,0.3], linewidth=0.5)
    plt.plot(p[:,6], p[:,7], '-', color=[0.3,0.3,0.3], linewidth=0.5)

    fi = p[-1,:].reshape(d,n,order='F')
    # final formation edges

    edges = [(0,1),(0,2),(1,2),(0,3),(2,3)]
    plt.plot([fi[0,0], fi[0,1]], [fi[1,0], fi[1,1]], 'b-', linewidth=0.5)
    plt.plot([fi[0,0], fi[0,2]], [fi[1,0], fi[1,2]], 'b-', linewidth=0.5)
    plt.plot([fi[0,1], fi[0,2]], [fi[1,1], fi[1,2]], 'b-', linewidth=0.5)
    plt.plot([fi[0,0], fi[0,3]], [fi[1,0], fi[1,3]], 'b-', linewidth=0.5)
    plt.plot([fi[0,2], fi[0,3]], [fi[1,2], fi[1,3]], 'b-', linewidth=0.5)

    pinit = p[0,:].reshape(d,n,order='F')
    plt.plot([pinit[0,0], pinit[0,1]], [pinit[1,0], pinit[1,1]], 'b-', linewidth=0.5)
    plt.plot([pinit[0,0], pinit[0,2]], [pinit[1,0], pinit[1,2]], 'b-', linewidth=0.5)
    plt.plot([pinit[0,1], pinit[0,2]], [pinit[1,1], pinit[1,2]], 'b-', linewidth=0.5)
    plt.plot([pinit[0,0], pinit[0,3]], [pinit[1,0], pinit[1,3]], 'b-', linewidth=0.5)
    plt.plot([pinit[0,2], pinit[0,3]], [pinit[1,2], pinit[1,3]], 'b-', linewidth=0.5)

    # draw robots at initial and final
    for i in range(n):
        draw_unicycle([pinit[0,i], pinit[1,i], theta[0,i]], scale)
        plt.text(pinit[0,i], pinit[1,i], str(i+1), color='w',
                 ha='center', fontsize=8, fontweight='bold')
        draw_unicycle([fi[0,i], fi[1,i], theta[-1,i]], scale)
        plt.text(fi[0,i], fi[1,i], str(i+1), color='w',
                 ha='center', fontsize=8, fontweight='bold')

    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.axis("equal")
    plt.legend(["traj1","traj2","traj3","traj4"])
    plt.grid(True)

    # --- 2. δ3, δ4 ---
    plt.figure()
    plt.plot(t, v[:,2], 'r-', linewidth=0.5, label=r'$\delta_3$')
    plt.plot(t, v[:,3], 'b--', linewidth=0.5, label=r'$\delta_4$')
    plt.xlabel("Time [s]")
    plt.ylabel(r"$\delta_i = ||v_i - v_c||$")
    plt.legend()
    plt.grid(True)

    # --- 3. ||g - g*|| ---
    plt.figure()
    plt.plot(t, e_g, 'r-', linewidth=1)
    plt.xlabel("Time [s]")
    plt.ylabel(r"$||g - g^*||$")
    plt.ylim([0,1])
    plt.grid(True)

    # --- 4. (Optional) Video ---
    if save_video:
        import matplotlib.animation as animation
        fig, (ax1, ax2) = plt.subplots(2,1, figsize=(6,8))

        def update(frame):
            ax1.clear(); ax2.clear()
            fi = p[frame,:].reshape(d,n,order='F')

            # edges
            
            ax1.plot([fi[0,0], fi[0,1]], [fi[1,0], fi[1,1]], 'b-', linewidth=1)
            ax1.plot([fi[0,0], fi[0,2]], [fi[1,0], fi[1,2]], 'b-', linewidth=1)
            ax1.plot([fi[0,1], fi[0,2]], [fi[1,1], fi[1,2]], 'b-', linewidth=1)
            ax1.plot([fi[0,0], fi[0,3]], [fi[1,0], fi[1,3]], 'b-', linewidth=1)
            ax1.plot([fi[0,2], fi[0,3]], [fi[1,2], fi[1,3]], 'b-', linewidth=1)


            # trajectories up to frame
            ax1.plot(p[:frame+1, 0], p[:frame+1, 1], '-', color=[0.3,0.3,0.3], linewidth=1)
            ax1.plot(p[:frame+1, 2], p[:frame+1, 3], '-', color=[0.3,0.3,0.3], linewidth=1)
            ax1.plot(p[:frame+1, 4], p[:frame+1, 5], '-', color=[0.3,0.3,0.3], linewidth=1)
            ax1.plot(p[:frame+1, 6], p[:frame+1, 7], '-', color=[0.3,0.3,0.3], linewidth=1)

            for i in range(n):
                pi = p[frame, 0:d*n].reshape(d,n,order='F')
                draw_unicycle([pi[0,i], pi[1,i], theta[frame,i]], scale)
                ax1.text(pi[0,i], pi[1,i], str(i), color='w', fontsize=8, ha='center', va='center', fontweight='bold')
                
                if frame == N - 1 :
                    pinit = p[0,:].reshape(d,n,order='F')
                    draw_unicycle([pinit[0,i], pinit[1,i], theta[0,i]], scale)
                    ax1.text(pinit[0,i], pinit[1,i], str(i), color='w', fontsize=8, ha='center', va='center', fontweight='bold')
                             
            ax1.set_xlim([-5,45])
            ax1.set_ylim([-3,30])
            ax1.set_title("Robot Formation")

            # error subplot
            ax2.plot(t[:frame+1], e_g[:frame+1], 'r-', linewidth=1.5)
            ax2.set_xlim([0,T]); ax2.set_ylim([0,1])
            ax2.set_ylabel(r"$||g-g^*||$")
            ax2.set_xlabel("Time [s]")

        ani = animation.FuncAnimation(fig, update, frames=range(0,N,4), interval=100)
        ani.save("RobotFormation.mp4", fps=10, dpi=150)

    plt.show()
