import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

def load_leader_trajectories(csv_file, dt=0.1):
    data = pd.read_csv(csv_file)

    t_data = np.arange(len(data)) * dt

    p1_traj = data[["p1_x", "p1_y"]].to_numpy()
    p2_traj = data[["p2_x", "p2_y"]].to_numpy()

    v1_traj = data[["p1_dot_x", "p1_dot_y"]].to_numpy()
    v2_traj = data[["p2_dot_x", "p2_dot_y"]].to_numpy()

    a1_traj = np.gradient(v1_traj, dt, axis=0)
    a2_traj = np.gradient(v2_traj, dt, axis=0)
    
    p1_func = interp1d(t_data, p1_traj, axis=0, fill_value="extrapolate")
    p2_func = interp1d(t_data, p2_traj, axis=0, fill_value="extrapolate")
    v1_func = interp1d(t_data, v1_traj, axis=0, fill_value="extrapolate")
    v2_func = interp1d(t_data, v2_traj, axis=0, fill_value="extrapolate")
    a1_func = interp1d(t_data, a1_traj, axis=0, fill_value="extrapolate")
    a2_func = interp1d(t_data, a2_traj, axis=0, fill_value="extrapolate")

    return p1_func, p2_func, v1_func, v2_func, a1_func, a2_func