from dsmpy.seismicmodel import SeismicModel
from dsmpy.modelparameters import ModelParameters, ParameterType
from dsmpy.windowmaker import WindowMaker
from dsmpy.window import Window
from dsmpy.component import Component
from dsmpy.utils.cmtcatalog import read_catalog
from dsmpy.utils.sklearnutils import get_XY
from dsmpy.dataset import Dataset, filter_sac_files, read_sac_meta
from dsmpy.dataset import read_sac_from_windows
from dsmpy.dataset import get_station, get_event_id
from dsmpy.dsm import compute_models_parallel
from pytomo.work.ca.params import get_dataset, get_model_syntest1_prem_vshvsv
from pytomo.utilities import get_temporary_str
from pytomo.inversion.fwi import FWI, freqency_hash, frequencies_from_hash
import matplotlib.pyplot as plt
from mpi4py import MPI
import logging
import glob
import collections

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
        glob.iglob('/work/anselme/central_pac/DATA/DATA/tmp/20*/*T'))
    # sac_files = list(
    #    glob.iglob('/Users/navy/git/dsmpy/tests/sac_files_2/*T')
    # )

    catalog = read_catalog()
    sac_meta, traces = read_sac_meta(sac_files)
    traces_filt = [tr for tr, meta in zip(traces, sac_meta)
                      if (
                        meta['evcount'] >= 30
                        and meta['stcount'] >= 5
                        and -170 <= meta['stlo'] <= -150
                        and (meta['evlo'] > 175 or meta['evlo'] < -170)
                        and meta['evdp'] >= 150
                        and meta['evnm'] in catalog
                      )
                      ]
    logging.info(f'Number of pre-selected SAC files: {len(traces_filt)}')

    t_before = 20.
    t_after = 40.
    buffer = 10.

    windows = WindowMaker.windows_from_obspy_traces(
        traces_filt, 'prem', ['S', 'ScS', 'sS'], [Component.T],
        t_before=t_before, t_after=t_after
    )
    windows = [window for window in windows
               if 70 <= window.get_epicentral_distance() <= 80]

    windows_ScS = []
    windows_S = []
    windows_S_sS = []
    for w in windows:
        if w.phase_name == 'ScS':
            windows_ScS.append(w)
        elif w.phase_name == 'S':
            windows_S.append(w)
            windows_S_sS.append(w)
        elif w.phase_name == 'sS':
            windows_S_sS.append(w)

    windows_S_sS_trim = WindowMaker.set_limit(
        windows_S_sS, t_before=5, t_after=15, inplace=False)
    windows_ScS_trimmed = WindowMaker.trim_windows(
        windows_ScS, windows_S_sS_trim)
    windows_proc = windows_ScS_trimmed + windows_S

    # set a buffer for shifting windows using the reference phase
    WindowMaker.set_limit(windows_proc, t_before - buffer, t_after + buffer)

    logging.info(
        f'Number of processed ScS windows: '
        f'{len([w for w in windows_proc if w.phase_name == "ScS"])}')

    dataset_dict = dict()
    windows_dict = dict()
    for freq, freq2 in [[0.01, 0.05], [0.01, 0.08]]:
        ds = Dataset.dataset_from_sac_process(
                sac_files, windows_proc, freq, freq2,
                filter_type='bandpass', shift=False)
        freq_hash = freqency_hash(freq, freq2)
        dataset_dict[freq_hash] = ds
        windows_dict[freq_hash] = windows_proc

    logging.info('Starting the inversion...')
    fwi = FWI(
        model_ref, model_params, dataset_dict,
        windows_dict, n_phases=1, mode=mode, phase_ref='S', buffer=buffer)
    fwi.set_selection(var=2.5, corr=0., ratio=2.5)

    model_1 = fwi.step(model_ref, 0.01, 0.05, n_pca_components=[2, 4], alphas=[.8, 1.])
    model_2 = fwi.step(model_1, 0.01, 0.05, n_pca_components=[2, 4], alphas=[.8, 1.])
    model_3 = fwi.step(model_2, 0.01, 0.08, n_pca_components=[4, 6], alphas=[.8, 1.])
    model_4 = fwi.step(model_3, 0.01, 0.08, n_pca_components=[4, 6], alphas=[.8, 1.])
    # model_5 = fwi.step(model_4, 0.01, 0.08, n_pca_components=[12, 16, 20, 24])

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
        figname = f'model{get_temporary_str()}.pdf'
        fig.savefig(figname, bbox_inches='tight')
        plt.close(fig)
