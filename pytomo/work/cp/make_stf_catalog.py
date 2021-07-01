from dsmpy.dataset import Dataset
from dsmpy.seismicmodel import SeismicModel
from dsmpy.windowmaker import WindowMaker
from dsmpy.component import Component
from pytomo.preproc.stream import sac_files_iterator
from pytomo.preproc.stfgridsearch import STFGridSearch
from mpi4py import MPI
import numpy as np
import logging

logging.basicConfig(
        level=logging.INFO, filename='makestfcatalog.log', filemode='w')

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    model = SeismicModel.prem()
    tlen = 1638.4
    nspc = 256
    sampling_hz = 20
    freq = 0.005
    freq2 = 0.1
    duration_min = 1.
    duration_max = 15.
    duration_inc = .25
    amp = 1.6
    amp_inc = .02
    distance_min = 40.
    distance_max = 80.
    dir_syn = '.'
    t_before = 10.
    t_after = 20.
    buffer = 10.
    catalog_path = 'stf_catalog.txt'
    n_distinct_comp_phase = 1

    for sac_files in sac_files_iterator(
            '/work/anselme/central_pac/DATA/DATA/20*/*T'):
        logging.write('{} num sacs = {}\n'.format(rank, len(sac_files)))

        if rank == 0:
            logging.write('{} reading dataset\n'.format(rank))
            dataset = Dataset.dataset_from_sac(sac_files, headonly=False)

            logging.write('{} computing time windows\n'.format(rank))
            windows_S = WindowMaker.windows_from_dataset(
                dataset, 'prem', ['S'],
                [Component.T], t_before=t_before, t_after=t_after)
            windows = windows_S
            windows = [
                window for window in windows
                if (distance_min <= window.get_epicentral_distance()
                    <= distance_max)]
        else:
            dataset = None
            windows = None

        durations = np.linspace(
            duration_min, duration_max,
            int((duration_max - duration_min) / duration_inc) + 1,
            dtype=np.float32)
        amplitudes = np.linspace(1., amp, int((amp - 1) /amp_inc) + 1)
        amplitudes = np.concatenate((1. / amplitudes[:0:-1], amplitudes))

        logging.write('Init stfgrid; filter dataset')
        stfgrid = STFGridSearch(
            dataset, model, tlen, nspc, sampling_hz, freq, freq2, windows,
            durations, amplitudes, n_distinct_comp_phase, buffer)

        logging.write('{} computing synthetics\n'.format(rank))
        misfit_dict, count_dict = stfgrid.compute_parallel(
            mode=2, dir=dir_syn, verbose=2)

        if rank == 0:
            logging.write('{} saving misfits\n'.format(rank))
            best_params_dict = stfgrid.get_best_parameters(
                misfit_dict, count_dict)
            stfgrid.write_catalog(
                catalog_path, best_params_dict, count_dict)
            for event_id in misfit_dict.keys():
                filename = '{}.pdf'.format(event_id)
                logging.write(
                    '{} saving figure to {}\n'.format(rank, filename))
                stfgrid.savefig(best_params_dict, event_id, filename)
