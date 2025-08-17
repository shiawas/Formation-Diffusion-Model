import numpy as np

def dh_i(x,w):
    vec = np.array([-np.sin(x), np.cos(x)]).reshape(2,1,order='F')
    return vec * w