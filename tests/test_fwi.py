from dsmpy.seismicmodel import SeismicModel
from dsmpy.modelparameters import ModelParameters, ParameterType
from dsmpy.windowmaker import WindowMaker
from dsmpy.window import Window
from dsmpy.dataset import Dataset
from dsmpy.component import Component
from pytomo.work.ca.params import get_dataset, get_model_syntest1_prem_vshvsv
from pytomo.work.ca.params import get_model_syntest2_prem_vshqmu, get_ref_event2
from pytomo.work.ca.params import get_model_syntest3_prem_vshqmu
from pytomo.utilities import get_temporary_str
from pytomo.inversion.fwi import FWI, freqency_hash
import matplotlib.pyplot as plt
from mpi4py import MPI
import math

if __name__ == '__main__':
    types = [ParameterType.VSH, ParameterType.QMU]
    radii = [3480. + 20 * i for i in range(21)]
    # radii = [3480., 3880]
    model_params = ModelParameters(
        types=types,
        radii=radii,
        mesh_type='boxcar')
    model_ref = SeismicModel.prem().boxcar_mesh(model_params)

    t_before = 20.
    t_after = 40.
    sampling_hz = 20
    mode = 2
    freq = 0.01
    freq2 = 0.04
    phases = ['S', 'sS', 'ScS']
    buffer = 10.

    dataset, _ = get_dataset(
        get_model_syntest2_prem_vshqmu(), tlen=1638.4, nspc=512,
        sampling_hz=sampling_hz,
        mode=mode, add_noise=False, noise_normalized_std=1.)

    windows = WindowMaker.windows_from_dataset(
        dataset, 'prem', phases, [Component.T],
        t_before=t_before, t_after=t_after)

    windows_ScS = [w for w in windows if w.phase_name == 'ScS']
    windows_S = [w for w in windows if w.phase_name == 'S']
    windows_S_sS = [w for w in windows if w.phase_name in {'S', 'sS'}]
    windows_S_sS_trim = WindowMaker.set_limit(
        windows_S_sS, t_before=5, t_after=15, inplace=False)
    windows_ScS_trimmed = WindowMaker.trim_windows(
        windows_ScS, windows_S_sS_trim)
    WindowMaker.set_limit(
        windows_S, t_before=5, t_after=15)
    windows_proc = windows_ScS_trimmed + windows_S

    # set a buffer for shifting windows using the reference phase
    WindowMaker.set_limit(windows_proc, t_before + buffer, t_after + buffer)
    npts_max = max([
        math.ceil(w.get_length() * sampling_hz) for w in windows_proc])

    dataset_dict = dict()
    windows_dict = dict()
    for freq, freq2 in [[0.01, 0.05], [0.01, 0.08], [0.01, 0.125]]:
        ds = dataset.filter(freq, freq2, type='bandpass', inplace=False)
        ds.apply_windows(
            windows_proc, n_phase=2, npts_max=npts_max)
        freq_hash = freqency_hash(freq, freq2)
        dataset_dict[freq_hash] = ds
        windows_dict[freq_hash] = windows_proc

    fwi = FWI(
        model_ref, model_params, dataset_dict,
        windows_dict, n_phases=1, mode=mode, phase_ref='S', buffer=buffer)
    fwi.set_selection(var=2.5, corr=0, ratio=2.5)

    fwi.set_ignore_types([ParameterType.QMU])
    model_1 = fwi.step(
        model_ref, 0.01, 0.05, n_pca_components=[2, 4, 6], alphas=[.8, 1.])
    model_2 = fwi.step(
        model_1, 0.01, 0.08, n_pca_components=[4, 6, 8], alphas=[.8, 1.])

    fwi.set_ignore_types([ParameterType.QMU], ignore=False)
    model_3 = fwi.step(
        model_2, 0.01, 0.08, n_pca_components=[4, 6, 8], alphas=[1.])
    model_4 = fwi.step(
        model_3, 0.01, 0.125, n_pca_components=[8, 10, 12, 14],
        alphas=[1.])

    if MPI.COMM_WORLD.Get_rank() == 0:
        # save FWI result to object
        fname = 'fwiresult_' + get_temporary_str() + '.pkl'
        fwi.results.save(fname)

        # plot models
        fig, (ax1, ax2) = plt.subplots(1, 2)
        model_ref.plot(types=[types[0]], ax=ax1, label='prem')
        model_ref.plot(types=[types[1]], ax=ax2, label='prem')
        get_model_syntest1_prem_vshvsv().plot(
            types=[types[0]], ax=ax1, label='target')
        model_1.plot(types=[types[0]], ax=ax1, label='it1')
        model_2.plot(types=[types[0]], ax=ax1, label='it2')
        model_3.plot(types=[types[0]], ax=ax1, label='it3')
        model_4.plot(types=[types[0]], ax=ax1, label='it4')
        # model_5.plot(types=[types[0]], ax=ax1, label='it5')
        ax1.set_ylim([3480, 4000])
        ax1.set_xlim([6.5, 8])
        ax2.set_xlim([100, 800])
        ax1.legend()
        plt.show()