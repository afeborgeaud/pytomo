import numpy as np
import glob
from mpi4py import MPI
from dsmpy.dataset import Dataset
from dsmpy.seismicmodel import SeismicModel
from dsmpy.component import Component
from dsmpy.windowmaker import WindowMaker
from dsmpy.dsm import compute_dataset_parallel, compute_models_parallel

def compute_misfits(dataset, model, windows, mode=0):
    """Return a dict with misfits for each time windows.

    Args:
        dataset (Dataset): dataset
        model (SeismicModel): reference model
        windows (list of Window): time windows
        mode (int): computation mode.

    Returns:
        dict: {'frequency': [(int, int)],
                         'misfit': [{'variance': np.ndarray,
                                     'corr': np.ndarray,
                                     'ratio': np.ndarray}]}
    """
    tlen = 1638.4
    nspc = 256

    outputs = compute_models_parallel(
        dataset, [model], tlen=tlen,
        nspc=nspc, sampling_hz=dataset.sampling_hz, mode=mode)

    freqs = [0.01, 0.01]
    freqs2 = [0.04, 0.08]

    if MPI.COMM_WORLD.Get_rank() == 0:
        misfits = dict()
        misfits['frequency'] = list(zip(freqs, freqs2))
        misfits['misfit'] = []
        for ifreq, (freq, freq2) in enumerate(zip(freqs, freqs2)):
            ds = dataset.filter(freq, freq2, 'bandpass', inplace=False)
            ds.apply_windows(windows, 1, 60 * 20)
            variances = np.zeros(len(windows))
            corrs = np.zeros(len(windows))
            ratios = np.zeros(len(windows))
            for iev in range(len(outputs)):
                event = dataset.events[iev]
                output = outputs[0][iev]
                output.filter(freq, freq2, 'bandpass')
                start, end = ds.get_bounds_from_event_index(iev)
                for ista in range(start, end):
                    station = ds.stations[ista]
                    jsta = np.argwhere(output.stations == station)[0][0]
                    i_windows_filt = [
                        (i, window) for i, window in enumerate(windows)
                        if (window.station == station
                            and window.event == event)]
                    for i, (iwin, window) in enumerate(i_windows_filt):
                        window_arr = window.to_array()
                        icomp = window.component.value
                        i_start = int(window_arr[0] * ds.sampling_hz)
                        i_end = int(window_arr[1] * ds.sampling_hz)
                        u_cut = output.us[icomp, jsta, i_start:i_end]
                        data_cut = ds.data[
                            i, icomp, ista, :]
                        residual = data_cut - u_cut
                        variance = np.var(residual)
                        if np.abs(data_cut).max() > 0:
                            variance /= np.abs(data_cut).max()**2
                        corr = np.corrcoef(data_cut, u_cut)[0, 1]
                        ratio = (data_cut.max() - data_cut.min()) / (
                            u_cut.max() - u_cut.min())
                        variances[iwin] = variance
                        corrs[iwin] = corr
                        ratios[iwin] = ratio
            misfits['misfit'].append(
                {'variance': variances, 'corr': corrs, 'ratio': ratios}
            )

            output.free()
    else:
        misfits = None

    return misfits


if __name__ == '__main__':
    sac_files = glob.glob(
        '/work/anselme/CA_ANEL_NEW/VERTICAL/200707211534A/*T')
    dataset = Dataset.dataset_from_sac(sac_files, headonly=False)
    model = SeismicModel.prem()
    tlen = 3276.8
    nspc = 1024
    sampling_hz = 20
    freq = 0.005
    freq2 = 0.167

    comm = MPI.COMM_WORLD
    n_cores = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        windows_S = WindowMaker.windows_from_dataset(
            dataset, 'prem', ['s', 'S', 'Sdiff'],
            [Component.T], t_before=10., t_after=20.)
        windows_P = WindowMaker.windows_from_dataset(
            dataset, 'prem', ['p', 'P', 'Pdiff'],
            [Component.Z], t_before=10., t_after=20.)
        windows = windows_S #+ windows_P

        dataset = Dataset.dataset_from_sac(sac_files, headonly=False)
    else:
        dataset = None

