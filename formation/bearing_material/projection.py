import numpy as np

n = 2 

def Pr(x):
    x = np.array(x).reshape(n,1,order='F')
    return np.eye(n) - ( x @ x.T ) / np.linalg.norm(x)**2