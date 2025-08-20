import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

def load_leader_trajectories(csv_file, dt=0.01):
    data = pd.read_csv(csv_file)

    t_data = np.arange(len(data)) * dt

    p1_traj = data[["p1_x", "p1_y"]].to_numpy()
    p2_traj = data[["p2_x", "p2_y"]].to_numpy()

    v1_traj = data[["p1_dot_x", "p1_dot_y"]].to_numpy()
    v2_traj = data[["p2_dot_x", "p2_dot_y"]].to_numpy()
    
    phi_traj = data["phi"].to_numpy()
    phi_dot_traj = data["phi_dot"].to_numpy()

    p1_func = interp1d(t_data, p1_traj, axis=0, fill_value="extrapolate")
    p2_func = interp1d(t_data, p2_traj, axis=0, fill_value="extrapolate")
    v1_func = interp1d(t_data, v1_traj, axis=0, fill_value="extrapolate")
    v2_func = interp1d(t_data, v2_traj, axis=0, fill_value="extrapolate")
    phi_func = interp1d(t_data, phi_traj, axis=0, fill_value="extrapolate")
    phi_dot_func = interp1d(t_data, phi_dot_traj, axis=0, fill_value="extrapolate")


    return p1_func, p2_func, v1_func, v2_func, phi_func, phi_dot_func