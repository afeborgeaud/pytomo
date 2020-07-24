from obspy import read
import os
import glob

def read_sac(sacpaths_regex):
    """Read sac files
    Args:
        sacpaths_regex: regex specifies list of sac paths
    Returns:
        traces (obspy.traces): waveform traces
    """
    fullpath = os.path.expanduser(sacpaths_regex)
    traces = [read(sac_file)[0]
                for sac_file in
                glob.iglob(fullpath)]
    return traces