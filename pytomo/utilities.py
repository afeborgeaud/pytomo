import numpy as np

def white_noise(std, size):
    return np.random.normal(0, std, size)