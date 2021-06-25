import numpy as np
from datetime import datetime

def white_noise(std, size) -> np.ndarray:
    """Compute random gaussian noise"""
    return np.random.normal(0, std, size)

def get_temporary_str() -> str:
    """To use to name temporary files or folders"""
    return datetime.now().strftime('%Y%m%d%H%M%S')