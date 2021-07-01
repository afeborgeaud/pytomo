from dsmpy.dsm import compute_dataset_parallel
from dsmpy.dataset import Dataset
from dsmpy.seismicmodel import SeismicModel
from dsmpy.windowmaker import WindowMaker
from dsmpy.component import Component
from dsmpy.spc.stf import SourceTimeFunction
from dsmpy.dsm import PyDSMOutput
from dsmpy.spc.stfcatalog import STFCatalog
from pytomo.preproc.iterstack import find_best_shift
from pytomo.preproc.stream import sac_files_iterator
import os
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import logging

logging.basicConfig(
        level=logging.INFO, filename='stfgridsearch.log', filemode='w')


class STFGridSearch():
    """Compute triangular source time functions by grid search.
    Args:

    """
    def __init__(
            self, dataset, seismic_model, tlen, nspc, sampling_hz,
            freq, freq2, windows, durations, amplitudes,
            n_distinct_comp_phase, buffer=10.):
        """

        Args:
            dataset (Dataset):
            seismic_model (SeismicModel):
            tlen (float):
            nspc (int):
            sampling_hz (int):
            freq (float):
            freq2 (float):
            windows (list of Window):
            durations (float):
            amplitudes (float):
            n_distinct_comp_phase (int):
            buffer (int):
        """
        self.dataset = dataset
        self.seismic_model = seismic_model
        self.tlen = tlen
        self.nspc = nspc
        self.sampling_hz = sampling_hz
        self.freq = freq
        self.freq2 = freq2
        self.windows = windows
        self.durations = durations
        self.amplitudes = amplitudes
        self.buffer = buffer
        self.n_distinct_comp_phase = n_distinct_comp_phase
        
        if dataset is not None:
            self.dataset.filter(
                freq, freq2, type='bandpass', zerophase=False)

            t_before = windows[0].t_before
            t_after = windows[0].t_after
            window_npts_max = int((t_before + t_after) * sampling_hz)
            self.dataset.apply_windows(
                windows, n_distinct_comp_phase, window_npts_max, buffer)

    def load_outputs(
            self, dir=None, mode=0, verbose=0):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        if rank == 0:
            filenames = [
                os.path.join(dir, event.event_id+'.pkl')
                for event in self.dataset.events]
            all_found = np.array([os.path.isfile(f) for f in filenames]).all()
 
            if all_found:
                logging.info(
                    '{} loading pydsmoutputs from files\n'
                    .format(rank))
                outputs = [
                    PyDSMOutput.load(filename) for filename in filenames]
        else:
            outputs = None
            all_found = None
        all_found = comm.bcast(all_found, root=0)
        logging.info('{} all_found={}\n'.format(rank, all_found))

        if not all_found:
            logging.info('{} computing outputs\n'.format(rank))
            outputs = compute_dataset_parallel(
                self.dataset, self.seismic_model, self.tlen,
                self.nspc, self.sampling_hz, mode=mode,
                verbose=verbose)
            logging.info('{} done!\n'.format(rank))
        return outputs, all_found
    
    def compute_parallel(
            self, mode=0, dir=None, verbose=0):
        """Compute using MPI.

        Args:
            mode (int): 0: sh+psv, 1: psv, 2:sh
            dir (str):
            verbose (int): 0: quiet, 1: debug

        Returns:
            misfit_dict (dict): dict with event_id as key and
                a=np.ndarray((n_durations,n_amplitudes,3)) as values.
                a[:,:,0] gives durations; a[:,:,1] gives amplitudes;
                a[:,:,2] gives misfit values

        """
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        outputs, read_from_file = self.load_outputs(
            dir=dir, mode=mode, verbose=verbose)
        self.outputs = outputs
        if rank == 0:
            if not read_from_file:
                for output in outputs:
                    filename = output.event.event_id + '.pkl'
                    output.save(filename)
        
        if rank == 0:
            outputs_scat = []
            data_scat = []
            station_scat = []
            n = int(len(outputs) / size)
            for i in range(size):
                i0 = i*n
                if i < size - 1:
                    i1 = i0 + n
                else:
                    i1 = len(outputs)
                outputs_scat.append(outputs[i0:i1])
                
                data_ = []
                stations_ = []
                for j in range(i0, i1):
                    # TODO understand why the order of stations in dataset
                    # change when using more than one cpu
                    start, end = self.dataset.get_bounds_from_event_index(j)
                    event_data = np.array(self.dataset.data[:, :, start:end, :])
                    event_stations = np.array(self.dataset.stations[start:end])
                    data_.append(event_data)
                    stations_.append(event_stations)
                data_scat.append(data_)
                station_scat.append(stations_)
        else:
            outputs_scat = None
            data_scat = None
            station_scat = None
        outputs_local = comm.scatter(outputs_scat, root=0)
        data_local = comm.scatter(data_scat, root=0)
        station_local = comm.scatter(station_scat, root=0)
        windows_local = comm.bcast(self.windows, root=0)

        logging.info('{} computing misfits\n'.format(rank))
        logging.info(
            '{} len(outputs_local)={}\n'.format(rank, len(outputs_local)))
        misfit_dict_local = dict()
        count_dict_local = dict()

        for iev, output in enumerate(outputs_local):
            event = output.event
            event_misfits = np.zeros(
                (len(self.durations), len(self.amplitudes), 3),
                dtype=np.float32)
            count_used_windows = np.zeros(
                (len(self.durations), 2),
                dtype=np.float32)
            event_misfits[:, :, 2] = 0.
            for i in range(len(self.durations)):
                event_misfits[i, :, 0] = self.durations[i]
                count_used_windows[i, 0] = self.durations[i]
            for i in range(len(self.amplitudes)):
                event_misfits[:, i, 1] = self.amplitudes[i]

            for idur, duration in enumerate(self.durations):
                logging.info('{} {}\n'.format(rank, idur))
                stf = SourceTimeFunction('triangle', duration/2.)
                output.to_time_domain(stf)
                output.filter(
                    self.freq, self.freq2,
                    type='bandpass', zerophase=False)
                
                for ista in range(len(output.stations)):
                    station = output.stations[ista]
                    windows_filt = [
                        window for window in windows_local
                        if (window.station == station
                            and window.event == event)]

                    jsta = np.argwhere(station_local[iev]==station)[0][0]
                    
                    for iwin, window in enumerate(windows_filt):
                        u_cut, data_cut = self.cut_data(
                            output.sampling_hz, output.us,
                            data_local[iev], window, iwin, ista, jsta)
                        
                        keep_data = self.select_data(data_cut, u_cut)
                        if keep_data:
                            for iamp, amp in enumerate(self.amplitudes):
                                misfit = STFGridSearch._misfit(
                                    data_cut, u_cut * amp)
                                if np.isnan(misfit):
                                    logging.info(
                                        '{} NaN misfit {} {}\n'
                                        .format(rank, station, event))
                                    continue
                                event_misfits[idur, iamp, 2] += misfit
                            count_used_windows[idur, 1] += 1

                if count_used_windows[idur, 1] > 0:
                    event_misfits[idur, :, 2] /= count_used_windows[idur, 1]
                else:
                    event_misfits[idur, :, 2] = 1e10
            misfit_dict_local[output.event.event_id] = event_misfits
            count_dict_local[output.event.event_id] = count_used_windows

        misfit_dict_list = comm.gather(misfit_dict_local, root=0)
        count_dict_list = comm.gather(count_dict_local, root=0)

        if rank == 0:
            misfit_dict = misfit_dict_list[0]
            for d in misfit_dict_list[1:]:
                misfit_dict.update(d)

            count_dict = count_dict_list[0]
            for d in count_dict_list[1:]:
                count_dict.update(d)
        else:
            misfit_dict = None
            count_dict = None

        return misfit_dict, count_dict

    def cut_data(
            self, sampling_hz, syn, data_local,
            window, iwin, ista, jsta):
        window_arr = window.to_array()
        icomp = window.component.value
        i_start = int(window_arr[0] * sampling_hz)
        i_end = int(window_arr[1] * sampling_hz)
        u_cut = syn[icomp, ista, i_start:i_end]
        
        data_cut_tmp = data_local[
            iwin, icomp, jsta]
        shift = 0
        try:
            shift, _ = find_best_shift(
                data_cut_tmp, u_cut,
                shift_polarity=False,
                skip_freq=1)
        except:
            print('Problem with finding best shift')

        data_cut = data_local[
            iwin, icomp, jsta, shift:(i_end-i_start+shift)]

        return u_cut, data_cut

    def select_data(self, obs, syn):
        if (np.isnan(obs).any()
            or syn.max() == 0.):
            return False
        amp_ratio_ref = (
            (obs.max() - obs.min())
            / (syn.max() - syn.min()))
        corr_ref = np.corrcoef(obs, syn)[0, 1]
        if (amp_ratio_ref > 2.5
                or amp_ratio_ref < 1/2.5
                or corr_ref < 0.5):
            return False
        else:
            return True

    def get_best_parameters(
            self, misfit_dict, count_dict, duration_cap=1.3):
        """Get best duration and amplitude correction from misfit_dict.

        Args:
            misfit_dict (dict): dict returned by compute_parallel()
            count_dict (dict): dict returbed by compute_parallel()

        Returns:
            dict: best_params_dict. keys are event_id; values are
                tuples (duration, amplitude).

        """
        best_params_dict = dict()
        for event_id in misfit_dict.keys():
            misfits = misfit_dict[event_id]

            # use only measurements with large num. of windows
            counts = count_dict[event_id]
            threshold = counts[:,1].max() / np.sqrt(2)
            mask = counts[:,1] < threshold
            misfits[mask, :, 2] = 1e10

            # remove measurements with duration larger
            # than the GCMT duration * duration_cap
            duration_gcmt = [
                    event.source_time_function.half_duration * 2. 
                    for event in self.dataset.events
                    if event.event_id==event_id][0]
            mask = self.durations > duration_gcmt*duration_cap
            misfits[mask, :, 2] = 1e10

            min_misfit = misfits[:, :, 2].min()
            dur, amp, _ = misfits[misfits[:,:,2]==min_misfit][0]

            best_params_dict[event_id] = (dur, amp)
        return best_params_dict

    def write_catalog(
            self, filename, best_params_dict, count_dict):
        with open(filename, 'a') as f:
            f.write('event_id duration amp_corr num_win duration_gcmt\n')
            for event_id in best_params_dict.keys():
                duration, amp_corr = best_params_dict[event_id]
                n_windows = count_dict[event_id]
                n_window = n_windows[
                    np.abs(n_windows[:, 0] - duration) < 1e-5][0, 1]
                duration_gcmt = [
                    event.source_time_function.half_duration * 2
                    for event in self.dataset.events
                    if event.event_id == event_id][0]
                line = STFCatalog._format_line(
                    event_id, duration, amp_corr, n_window, duration_gcmt)
                f.write(line + '\n')
    
    def savefig(self, best_params_dict, event_id, filename):
        output = [
            out for out in self.outputs if out.event.event_id==event_id][0]
        iev = [
            i for i in range(len(self.outputs))
            if self.outputs[i].event.event_id==event_id][0]

        duration, amp = best_params_dict[event_id]
        stf = SourceTimeFunction('triangle', duration/2.)

        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

        output.to_time_domain(stf)
        output.filter(
            self.freq, self.freq2,
            type='bandpass', zerophase=False)
        us = np.array(output.us)

        # gcmt stf
        output.to_time_domain()
        output.filter(
            self.freq, self.freq2,
            type='bandpass', zerophase=False)
        us_gcmt = np.array(output.us)

        start, end = self.dataset.get_bounds_from_event_index(iev)
        event = output.event
        for ista in range(len(output.stations)):
            station = output.stations[ista]
            windows_filt = [
                window for window in self.windows
                if (window.station == station
                    and window.event == event)]
            jsta = np.where(
                self.dataset.stations[start:end]==station)[0][0] + start
            for iwin, window in enumerate(windows_filt):
                u_cut, data_cut = self.cut_data(
                            output.sampling_hz, us, self.dataset.data,
                            window, iwin, ista, jsta)
                u_gcmt_cut, _ = self.cut_data(
                            output.sampling_hz, us_gcmt, self.dataset.data,
                            window, iwin, ista, jsta)

                keep_data = self.select_data(data_cut, u_cut)

                if keep_data:
                    distance = window.get_epicentral_distance()
                    if -data_cut.min() > data_cut.max():
                        norm = data_cut.min() * 2. # reverse polarity
                    else:
                        norm = data_cut.max() * 2.
                    ts = np.linspace(
                        0, len(data_cut)/output.sampling_hz, len(data_cut))
                    ax1.plot(ts, data_cut/norm+distance, color='black')
                    ax1.plot(ts, amp*u_cut/norm+distance, color='red')
                    ax2.plot(ts, data_cut/norm+distance, color='black')
                    ax2.plot(ts, u_gcmt_cut/norm+distance, color='green')

        ax1.title.set_text('Ours')
        ax2.title.set_text('GCMT')
        ax1.set(
            ylabel='Distance (deg)',
            xlabel='Time (s)')
        ax2.set(xlabel='Time (s)')
        plt.suptitle(event_id)

        plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)

    @staticmethod
    def _misfit(obs, syn):
        corr = np.corrcoef(obs, syn)[0, 1]
        amp_ratio = (obs.max() - obs.min()) / (syn.max() - syn.min())
        var = np.dot((obs-syn), (obs-syn)) / np.dot(obs, obs)
        return (0.5*(1 - corr)
                + 1.5*np.abs(np.log(2.5)*np.log(amp_ratio))
                + var / 2.5)


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    model = SeismicModel.prem()
    tlen = 3276.8
    nspc = 512
    sampling_hz = 20
    freq = 0.005
    freq2 = 0.1
    duration_min = 1.
    duration_max = 15.
    duration_inc = .25
    amp = 1.6
    amp_inc = .02
    distance_min = 10.
    distance_max = 90.
    dir_syn = '.'
    t_before = 10.
    t_after = 20.
    buffer = 10.
    catalog_path = 'stf_catalog.txt'
    n_distinct_comp_phase = 1

    for sac_files in sac_files_iterator(
            '/home/anselme/Dropbox/Kenji/MTZ_JAPAN/DATA/20*/*T'):
        logging.info('{} num sacs = {}\n'.format(rank, len(sac_files)))

        if rank == 0:
            logging.info('{} reading dataset\n'.format(rank))
            dataset = Dataset.dataset_from_sac(sac_files, headonly=False)

            logging.info('{} computing time windows\n'.format(rank))
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
                int((duration_max - duration_min)/duration_inc)+1,
                dtype=np.float32)
        amplitudes = np.linspace(1., amp, int((amp - 1) / amp_inc) + 1)
        amplitudes = np.concatenate((1. / amplitudes[:0:-1], amplitudes))

        logging.info('Init stfgrid; filter dataset')
        stfgrid = STFGridSearch(
            dataset, model, tlen, nspc, sampling_hz, freq, freq2, windows,
            durations, amplitudes, n_distinct_comp_phase, buffer)

        logging.info('{} computing synthetics\n'.format(rank))
        misfit_dict, count_dict = stfgrid.compute_parallel(
            comm, mode=2, dir=dir_syn, verbose=2)
        
        if rank == 0:
            logging.info('{} saving misfits\n'.format(rank))
            best_params_dict = stfgrid.get_best_parameters(
                misfit_dict, count_dict)
            stfgrid.write_catalog(
                catalog_path, best_params_dict, count_dict)
            for event_id in misfit_dict.keys():
                filename = '{}.pdf'.format(event_id)
                logging.info(
                    '{} saving figure to {}\n'.format(rank, filename))
                stfgrid.savefig(best_params_dict, event_id, filename)
