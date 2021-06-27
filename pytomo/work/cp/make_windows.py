import numpy as np
import glob
import sys
from mpi4py import MPI
from dsmpy.dataset import Dataset
from dsmpy.seismicmodel import SeismicModel
from dsmpy.component import Component
from dsmpy.windowmaker import WindowMaker
from dsmpy.dsm import compute_dataset_parallel, compute_models_parallel
from pytomo.preproc.dataselection import compute_misfits
import matplotlib.pyplot as plt


if __name__ == '__main__':
    sac_files = list(glob.iglob(sys.argv[1]))
    dataset = Dataset.dataset_from_sac(sac_files)
    windows = WindowMaker.windows_from_dataset(
        dataset, 'prem', ['ScS'], [Component.T],
        t_before=20, t_after=40)
    model = SeismicModel.prem()
    freqs = [0.01, 0.01]
    freqs2 = [0.04, 0.08]
    datasets = [
        Dataset.dataset_from_sac_process(
            sac_files, windows, freq, freq2)
        for freq, freq2 in zip(freqs, freqs2)
    ]
    misfits = compute_misfits(
        datasets, freqs, freqs2, model, windows, mode=0)

    print(misfits)

    if MPI.COMM_WORLD.Get_rank() == 0:
        selected_windows = dict()
        for ifreq in range(len(freqs)):
            windows_tmp = []
            for window, variance, corr, ratio in zip(
                windows, misfits['misfit'][0]['variance'],
                misfits['misfit'][0]['corr'],
                misfits['misfit'][0]['ratio'],
            ):
                if (
                    variance < 2.5
                    and corr > 0
                    and 0.4 < ratio < 2.5
                ):
                    windows_tmp.append(window)

            print(f'{len(windows)} windows\n'
                  f'{len(windows_tmp)} windows after selection')
        key = f'{freqs[ifreq]}_{freqs2[ifreq]}'
        selected_windows[key] = windows_tmp

        WindowMaker.save('windows.pkl', windows)
        WindowMaker.save('selected_windows.pkl', selected_windows)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        for i, (freq, freq2) in enumerate(misfits['frequency']):
            axes[i, 0].hist(misfits['misfit'][i]['variance'],
                            label='variance', bins=30)
            axes[i, 1].hist(misfits['misfit'][i]['corr'], label='corr',
                            bins=30)
            axes[i, 2].hist(misfits['misfit'][i]['ratio'], label='ratio',
                            bins=30)
        for ax in axes.ravel():
            ax.legend()
        plt.show()