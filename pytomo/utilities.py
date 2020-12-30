import numpy as np
from datetime import datetime

def white_noise(std, size):
    return np.random.normal(0, std, size)

def get_temporary_str():
    return datetime.now().strftime('%Y%m%d%H%M%S')