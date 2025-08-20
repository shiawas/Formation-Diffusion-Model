import numpy as np
import pandas as pd

def estimate_xi_v_from_csv(csv_path, dt):
    df = pd.read_csv(csv_path)

    v1 = df[['p1_dot_x', 'p1_dot_y']].to_numpy()  # (N,2)
    v2 = df[['p2_dot_x', 'p2_dot_y']].to_numpy()

    v1_norm = np.linalg.norm(v1, axis=1)
    v2_norm = np.linalg.norm(v2, axis=1)

    xi_a_max = float(max(v1_norm.max(), v2_norm.max()))

    return xi_a_max