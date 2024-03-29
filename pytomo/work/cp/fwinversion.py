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

logging.basicConfig(
    level=logging.INFO, filename='fwinversion.log', filemode='w')

if __name__ == '__main__':
    # type of mode parameters for inversion
    types = [ParameterType.VSH]

    # radii of layers to parametrize the inversion model
    radii = [3480. + 20 * i for i in range(21)]

    # define model parameter object for a boxcar mesh
    # (constant-property layers)
    model_params = ModelParameters(
        types=types,
        radii=radii,
        mesh_type='boxcar')

    # let the reference model be prem
    model_ref = SeismicModel.prem().boxcar_mesh(model_params)

    # computation mode. mode = 2 computes only the SH wavefield
    mode = 2

    # list of paths to SAC files used as data for the inversion
    sac_files = list(
        glob.iglob('/work/anselme/central_pac/DATA/DATA/tmp/20*/*T'))

    # read the GCMT catalog
    catalog = read_catalog()

    # read metadata on the SAC files, and filter events/stations
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

    # define the time windows to cut the data
    t_before = 20.
    t_after = 40.
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

    # trim the ScS windows when they overlap with S or sS
    windows_S_sS_trim = WindowMaker.set_limit(
        windows_S_sS, t_before=5, t_after=15, inplace=False)
    windows_ScS_trimmed = WindowMaker.trim_windows(
        windows_ScS, windows_S_sS_trim)
    windows_proc = windows_ScS_trimmed + windows_S

    # set a buffer used to compute static corrections for ScS
    # using S as a reference phase
    buffer = 10.
    WindowMaker.set_limit(windows_proc, t_before + buffer, t_after + buffer)

    logging.info(
        f'Number of processed ScS windows: '
        f'{len([w for w in windows_proc if w.phase_name == "ScS"])}')

    # read, filter, and cut the data from the SAC files
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
        windows_dict, n_phases=2, mode=mode, phase_ref='S', buffer=buffer)

    # During the gradient descent, consider only the data that
    # satisfy the selection criterion.
    fwi.set_selection(var=2.5, corr=0., ratio=2.5)

    # Perform 5 iterations using first longer periods (20 s)
    # then shorter periods (12.5 s)
    model_1 = fwi.step(
        model_ref, 0.01, 0.05, n_pca_components=[2, 4], alphas=[.8, 1.])
    model_2 = fwi.step(
        model_1, 0.01, 0.05, n_pca_components=[2, 4], alphas=[.8, 1.])
    model_3 = fwi.step(
        model_2, 0.01, 0.08, n_pca_components=[4, 6], alphas=[.8, 1.])
    model_4 = fwi.step(
        model_3, 0.01, 0.08, n_pca_components=[4, 6], alphas=[.8, 1.])
    model_5 = fwi.step(
        model_4, 0.01, 0.08, n_pca_components=[12, 16, 20, 24])

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
        model_5.plot(types=types, ax=ax, label='it5')
        ax.set_ylim([3480, 4000])
        ax.set_xlim([6.5, 8])
        ax.legend()
        figname = f'model{get_temporary_str()}.pdf'
        fig.savefig(figname, bbox_inches='tight')
        plt.close(fig)
