import glob
from mpi4py import MPI
from dsmpy.dataset import Dataset, filter_sac_files
from dsmpy.seismicmodel import SeismicModel
from dsmpy.component import Component
from dsmpy.windowmaker import WindowMaker
from dsmpy.dsm import compute_dataset_parallel
from pytomo.preproc.dataselection import compute_misfits
import matplotlib.pyplot as plt


if __name__ == '__main__':
    sac_files = list(
        glob.iglob('sac_files/*T'))

    dataset = Dataset.dataset_from_sac(sac_files)
    windows = WindowMaker.windows_from_dataset(
        dataset, 'prem', ['S'], [Component.T],
        t_before=20, t_after=40)
    model = SeismicModel.prem()
    freqs = [0.01]
    freqs2 = [0.08]
    datasets = [
        Dataset.dataset_from_sac_process(
            sac_files, windows, freq, freq2)
        for freq, freq2 in zip(freqs, freqs2)
    ]
    misfits, windows_shift = compute_misfits(
        datasets, freqs, freqs2, model, windows, mode=2)

    outputs = compute_dataset_parallel(
        datasets[0], model, tlen=1638.4,
        nspc=256, sampling_hz=datasets[0].sampling_hz, mode=2)
    for output in outputs:
        output.filter(freqs[0], freqs2[0], 'bandpass')

    ds = Dataset.dataset_from_sac_process(
        sac_files, windows_shift[0], freqs[0], freqs2[0], shift=False)
    ds_shift = Dataset.dataset_from_sac_process(
        sac_files, windows_shift[0], freqs[0], freqs2[0])

    fig, axes = plt.subplots(1, 2)
    ds.plot_event(0, component=Component.T, ax=axes[0], color='black')
    ds_shift.plot_event(0, component=Component.T, ax=axes[0], color='blue')
    outputs[0].plot_component(
        Component.T, windows=windows_shift[0], ax=axes[0], color='red',
        align_zero=True
    )
    ds.plot_event(1, component=Component.T, ax=axes[1], color='black')
    ds_shift.plot_event(1, component=Component.T, ax=axes[1], color='blue')
    outputs[1].plot_component(
        Component.T, windows=windows_shift[0], ax=axes[1], color='red',
        align_zero=True
    )
    plt.show()

    if MPI.COMM_WORLD.Get_rank() == 0:
        selected_windows = dict()
        for ifreq in range(len(freqs)):
            windows_tmp = []
            for window, variance, corr, ratio in zip(
                windows_shift[ifreq], misfits['misfit'][ifreq]['variance'],
                misfits['misfit'][ifreq]['corr'],
                misfits['misfit'][ifreq]['ratio'],
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
        WindowMaker.save('selected_shift_windows.pkl', selected_windows)

        fig, axes = plt.subplots(1, 3, figsize=(15, 10))
        for i, (freq, freq2) in enumerate(misfits['frequency']):
            variances = misfits['misfit'][i]['variance']
            ratios = misfits['misfit'][i]['ratio']
            variances = variances[variances < 5]
            ratios = ratios[ratios < 10]
            axes[i, 0].hist(variances,
                            label='variance', bins=30)
            axes[i, 1].hist(misfits['misfit'][i]['corr'], label='corr',
                            bins=30)
            axes[i, 2].hist(ratios, label='ratio',
                            bins=30)
        for ax in axes.ravel():
            ax.legend()
        plt.show()