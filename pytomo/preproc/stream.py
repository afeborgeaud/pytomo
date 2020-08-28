from obspy import read
import os
import glob
from collections import defaultdict

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

def sac_files_iterator(sacpaths_regex, comm=None, log=None):
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        fullpath = os.path.expanduser(sacpaths_regex)
        sac_files = list(glob.iglob(fullpath))
        traces = [read(sac_file, headonly=True)[0]
                    for sac_file in
                    sac_files]
        event_trace_dict = defaultdict(list)
        for trace, sac_file in zip(traces, sac_files):
            event_id = trace.stats.sac.kevnm
            event_trace_dict[event_id].append(sac_file)
        event_ids = list(event_trace_dict.keys())
        log.write(
            '{} n_events={} size={}\n'
            .format(rank, len(event_ids), size))
        chunks = int(len(event_ids) / size)
        if len(event_ids) % size != 0:
            chunks += 1
    else:
        chunks = None
    chunks = comm.bcast(chunks, root=0)
    if log is not None:
        log.write('{} chunks={}'.format(rank, chunks))

    if rank > 0:
        for i in range(chunks):
            yield []
    else:
        for i in range(chunks):
            files = list()
            if i < chunks-1:
                for j in range(i*size, (i+1)*size):
                    files += event_trace_dict[event_ids[j]]
            else:
                for j in range(i*size, len(event_ids)):
                    files += event_trace_dict[event_ids[j]]
            yield files

if __name__ == '__main__':
    it = sac_files_iterator(
        '/work/anselme/DATA/CENTRAL_AMERICA/200[56]*/*T', 3)
    for sac_files in it:
        print(len(sac_files))