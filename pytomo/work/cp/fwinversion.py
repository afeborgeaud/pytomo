from dsmpy.seismicmodel import SeismicModel
from dsmpy.modelparameters import ModelParameters, ParameterType
from dsmpy.windowmaker import WindowMaker
from dsmpy.window import Window
from dsmpy.component import Component
from dsmpy.utils.sklearnutils import get_XY
from dsmpy.dataset import Dataset, filter_sac_files, read_sac_meta
from dsmpy.dataset import get_station, get_event_id
from dsmpy.dsm import compute_models_parallel
from pytomo.work.ca.params import get_dataset, get_model_syntest1_prem_vshvsv
from pytomo.utilities import get_temporary_str
from pytomo.inversion.fwi import FWI, freqency_hash, frequencies_from_hash
import matplotlib.pyplot as plt
from mpi4py import MPI
import logging
import glob

logging.basicConfig(
    level=logging.INFO, filename='fwinversion.log', filemode='w')


def filter_from_windows(eventid_station):
    filt_windows = [
        w for w in windows
        if (w.event.event_id == eventid_station[0]
            and w.station == eventid_station[1])
    ]
    return len(filt_windows) > 0

def filter_corridor(evenentid_station):
    event = catalog[catalog == evenentid_station[0]]
    station = evenentid_station[1]


if __name__ == '__main__':
    types = [ParameterType.VSH]
    radii = [3480. + 20 * i for i in range(21)]
    model_params = ModelParameters(
        types=types,
        radii=radii,
        mesh_type='boxcar')
    model_ref = SeismicModel.prem().boxcar_mesh(model_params)

    window_file = 'selected_shift_windows_ScS_S_sS.pkl'
    mode = 2

    sac_files = list(
        glob.iglob('/work/anselme/central_pac/DATA/DATA/20*/*T'))
    sac_files = list(
        glob.iglob('/Users/navy/git/dsmpy/tests/sac_files_2/*T')
    )

    sac_meta = read_sac_meta(sac_files)

    windows_dict = WindowMaker.load(window_file)
    windows_ScS_proc_dict = dict()
    for freq_hash, windows in windows_dict.items():
        # windows = [
        #     w for w in windows
        #     if len([
        #         tr for sac_file, tr in sac_meta
        #         if (w.event.event_id == get_event_id(tr)
        #             and w.station == get_station(tr))
        #     ]) > 0
        # ]

        windows_ScS = []
        windows_S = []
        windows_S_sS = []
        for w in windows:
            if w.phase_name == 'ScS' and w.component == Component.T:
                windows_ScS.append(w)
            elif w.phase_name == 'S' and w.component == Component.T:
                windows_S.append(w)
                windows_S_sS.append(w)
            elif w.phase_name == 'sS' and w.component == Component.T:
                windows_S_sS.append(w)

        windows_S_sS_trim = WindowMaker.set_limit(
            windows_S_sS, t_before=5, t_after=15, inplace=False)
        windows_ScS_trimmed = WindowMaker.trim_windows(
            windows_ScS, windows_S_sS_trim)

        windows_ScS_proc = []
        for w in windows_ScS_trimmed:
            window_S = [
                window for window in windows_S
                if (w.station == window.station
                    and w.event == window.event
                    and w.component == window.component)
            ]
            if len(window_S) == 1:
                # windows_ScS_proc.append(
                #     Window(w.travel_time, w.event, w.station,
                #            w.phase_name, w.component, w.t_before,
                #            w.t_after, windows_S[0].t_shift)
                # )
                windows_ScS_proc.append(
                    Window(w.travel_time, w.event, w.station,
                           w.phase_name, w.component, w.t_before,
                           w.t_after, w.t_shift)
                )
        windows_ScS_proc_dict[freq_hash] = windows_ScS_proc
        logging.info(
            f'Number of processed ScS windows: {len(windows_ScS_proc)}')

    # sac_meta_filt = [
    #     (sac_file, tr) for sac_file, tr in sac_meta
    #     if len(
    #         [
    #             w for w in windows_ScS_proc
    #             if (w.event.event_id == get_event_id(tr)
    #                 and w.station == get_station(tr))
    #         ]
    #     ) > 0
    # ]
    # sac_files_filt = [sac_file for sac_file, tr in sac_meta_filt]
    # logging.info(f'Total number of SAC files used {len(sac_files_filt)}')

    sac_files = [sac_file for sac_file, tr in sac_meta]

    dataset_dict = dict()
    for freq_hash, windows in windows_ScS_proc_dict.items():
        freq, freq2 = frequencies_from_hash(freq_hash)
        ds = Dataset.dataset_from_sac_process(
                sac_files, windows, freq, freq2,
                filter_type='bandpass', shift=True)
        dataset_dict[freq_hash] = ds

    logging.info('Starting the inversion...')
    fwi = FWI(
        model_ref, model_params, dataset_dict,
        windows_ScS_proc_dict, n_phases=1, mode=mode)

    model_1 = fwi.step(model_ref, 0.01, 0.08, n_pca_components=[8])
    model_2 = fwi.step(model_1, 0.01, 0.04, n_pca_components=[8])
    model_3 = fwi.step(model_2, 0.01, 0.08, n_pca_components=[8, 12, 16])
    model_4 = fwi.step(model_3, 0.01, 0.08, n_pca_components=[8, 12, 16])
    model_5 = fwi.step(model_4, 0.01, 0.08, n_pca_components=[12, 16, 20, 24])

    if MPI.COMM_WORLD.Get_rank() == 0:
        # save FWI result to object
        fname = 'fwiresult_' + get_temporary_str() + '.pkl'
        fwi.results.save(fname)

        # plot models
        fig, ax = plt.subplots(1)
        model_ref.plot(types=types, ax=ax, label='prem')
        get_model_syntest1_prem_vshvsv().plot(
            types=types, ax=ax, label='target')
        model_1.plot(types=types, ax=ax, label='it1')
        model_2.plot(types=types, ax=ax, label='it2')
        model_3.plot(types=types, ax=ax, label='it3')
        model_4.plot(types=types, ax=ax, label='it4')
        ax.set_ylim([3480, 4000])
        ax.set_xlim([6.5, 8])
        ax.legend()
        plt.show()