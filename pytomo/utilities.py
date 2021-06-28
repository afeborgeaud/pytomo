import numpy as np
from datetime import datetime

def white_noise(std, size) -> np.ndarray:
    """Compute random gaussian noise"""
    return np.random.normal(0, std, size)

def get_temporary_str() -> str:
    """To use to name temporary files or folders"""
    return datetime.now().strftime('%Y%m%d%H%M%S')

def minimum_tlen(windows) -> float:
    """Find the minimum tlen that includes the longest time window.

    Args:
        windows (list of Window): time windows

    Returns:
        float: the minimum length for synthetics in seconds

    """
    t_max = np.array(
        [window.to_array()[1] for window in windows]).max()
    n_min = np.ceil(np.log2(10 * t_max))
    return 2 ** n_min / 10.

def minimum_nspc(tlen: float, freq2: float) -> int:
    """Find the minimum nspc for accurate computation given freq2.

    Args:
        tlen: length of synthetics in seconds
        freq2: the maximum filtering frequency in Hz

    Returns:
        int: the minimum number of frequency point for computation
            of accurate synthetics
    """
    x = np.ceil(np.log2(1.4 * tlen * freq2))
    return int(2 ** x)