from pydsm.dsm import compute_dataset_parallel
from pydsm.dataset import Dataset
from pydsm.seismicmodel import SeismicModel
from pydsm.windowmaker import WindowMaker
from pydsm.component import Component
from pydsm.spc.spctime import SourceTimeFunction
from pydsm.dsm import PyDSMOutput
from pytomo.preproc.iterstack import IterStack
from pytomo.preproc.stream import sac_files_iterator
import glob
import os
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

class STFGridSearch():
    '''Compute triangular source time functions by grid search.
    Args:

    '''
    def __init__(
            self, dataset, seismic_model, tlen, nspc, sampling_hz,
            freq, freq2, windows, durations, amplitudes):
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

        if dataset is not None:
            self.dataset.filter(
                freq, freq2, type='bandpass', zerophase=False)

    def load_outputs(
            self, comm, dir=None, mode=0, verbose=0, log=None):
        rank = comm.Get_rank()
        if rank == 0:
            filenames = [
                os.path.join(dir, event.event_id+'.pkl')
                for event in self.dataset.events]
            all_found = np.array([os.path.isfile(f) for f in filenames]).all()
 
            if all_found:
                if log is not None:
                    log.write(
                        '{} loading pydsmoutputs from files\n'
                        .format(rank))
                outputs = [
                    PyDSMOutput.load(filename) for filename in filenames]
        else:
            outputs = None
            all_found = None
        all_found = comm.bcast(all_found, root=0)
        log.write('{} all_found={}\n'.format(rank, all_found))

        if not all_found:
            if log is not None:
                log.write('{} computing outputs\n'.format(rank))
            outputs = compute_dataset_parallel(
                self.dataset, self.seismic_model, self.tlen,
                self.nspc, self.sampling_hz, comm, mode=mode,
                verbose=verbose, log=log)
            if log is not None:
                log.write('{} done!\n'.format(rank))
        return outputs, all_found
    
    def compute_parallel(
            self, comm, mode=0, dir=None, verbose=0, log=None):
        '''Compute using MPI.
        Args:
            comm (mpi4py.COMM_WORLD): MPI communicator
            mode (int): 0: sh+psv, 1: psv, 2:sh
            verbose (int): 0: quiet, 1: debug
        Returns:
            misfit_dict (dict): dict with event_id as key and
                a=ndarray((n_durations,n_amplitudes,3)) as values.
                a[:,:,0] gives durations; a[:,:,1] gives amplitudes;
                a[:,:,2] gives misfit values
        '''
        rank = comm.Get_rank()
        outputs, read_from_file = self.load_outputs(
            comm, dir=dir, mode=mode, verbose=verbose, log=log)
        self.outputs = outputs
        if rank == 0:
            if not read_from_file:
                for output in outputs:
                    filename = output.event.event_id + '.pkl'
                    output.save(filename)

        if rank == 0:
            log.write('{} computing misfits\n'.format(rank))
            misfit_dict = dict()
            for iev, output in enumerate(outputs):
                start, end = dataset.get_bounds_from_event_index(iev) 
                event = output.event
                event_misfits = np.zeros(
                    (len(durations), len(self.amplitudes), 3),
                    dtype=np.float32)
                for i in range(len(durations)):
                    event_misfits[i, :, 0] = durations[i]
                for i in range(len(amplitudes)):
                    event_misfits[:, i, 1] = amplitudes[i]

                for idur, duration in enumerate(durations):
                    stf = SourceTimeFunction('triangle', duration/2.)
                    output.to_time_domain(stf)
                    output.filter(
                        self.freq, self.freq2,
                        type='bandpass', zerophase=False)
                        
                    for i in range(end-start):
                        station = dataset.stations[i+start]
                        windows_filt = [
                            window for window in windows
                            if (window.station == station
                                and window.event == event)]
                        for window in windows_filt:
                            window_arr = window.to_array()
                            icomp = window.component.value
                            i_start = int(window_arr[0] * dataset.sampling)
                            i_end = int(window_arr[1] * dataset.sampling)
                            u_cut = output.us[icomp, i, i_start:i_end]
                            #TODO align obs and syn
                            buffer = 10 * dataset.sampling
                            i_start_buff = i_start - buffer
                            i_end_buff = i_end + buffer
                            data_cut_tmp = dataset.data[
                                icomp, i, i_start_buff:i_end_buff]
                            shift, _ = IterStack.find_best_shift(
                                data_cut_tmp, u_cut, shift_polarity=False)
                            shift -= buffer

                            data_cut = dataset.data[
                                icomp, i, (i_start+shift):(i_end+shift)]

                            for iamp, amp in enumerate(self.amplitudes):
                                misfit = STFGridSearch._misfit(
                                    data_cut, u_cut * amp)
                                event_misfits[idur, iamp, 2] += misfit

                event_misfits[:, :, 2] /= (end - start + 1)
                misfit_dict[output.event.event_id] = event_misfits

            return misfit_dict

                    # fig, ax = output.plot_component(
                    #     Component.T, windows=self.windows,
                    #     align_zero=True, color='red')
                    # _, ax = dataset.plot_event(
                    #     0, windows=self.windows, align_zero=True,
                    #     component=Component.T, ax=ax, color='black')
                    # plt.show()

    def get_best_parameters(self, misfit_dict):
        '''Get best duration and amplitude correction from misfit_dict.
        Args:
            misfit_dict (dict): dict returned by compute_parallel()
        Returns:
            best_params_dict (dict): keys are event_id; values are
                tuples (duration, amplitude).
        '''
        best_params_dict = dict()
        for event_id in misfit_dict.keys():
            misfits = misfit_dict[event_id]
            min_misfit = misfits[:, :, 2].min()
            dur, amp, _ = misfits[misfits[:,:,2]==min_misfit][0]

            best_params_dict[event_id] = (dur, amp)
        return best_params_dict

    def save_catalog(self, filename, best_params_dict):
        with open(filename, 'a') as f:
            for event_id in best_params_dict.keys():
                duration, amp_corr = best_params_dict[event_id]
                f.write(
                    '{} {} {}\n'.format(event_id, duration, amp_corr))

    def savefig(self, best_params_dict, event_id, filename):
        output = [
            out for out in self.outputs if out.event.event_id==event_id][0]
        iev = [
            i for i in range(len(self.outputs))
            if self.outputs[i].event.event_id==event_id][0]

        duration, amp = best_params_dict[event_id]
        stf = SourceTimeFunction('triangle', duration/2.)

        plt.clf()
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

        start, end = dataset.get_bounds_from_event_index(iev)
        event = output.event
        for i in range(end-start):
            station = dataset.stations[i+start]
            windows_filt = [
                window for window in windows
                if (window.station == station
                    and window.event == event)]
            for window in windows_filt:
                window_arr = window.to_array()
                icomp = window.component.value
                i_start = int(window_arr[0] * dataset.sampling)
                i_end = int(window_arr[1] * dataset.sampling)

                u_cut = us[icomp, i, i_start:i_end] * amp
                u_gcmt_cut = us_gcmt[icomp, i, i_start:i_end]

                #TODO align obs and syn
                buffer = 10 * dataset.sampling
                i_start_buff = i_start - buffer
                i_end_buff = i_end + buffer
                data_cut_tmp = dataset.data[
                    icomp, i, i_start_buff:i_end_buff]
                shift, _ = IterStack.find_best_shift(
                    data_cut_tmp, u_cut, shift_polarity=False)
                shift -= buffer

                data_cut = dataset.data[
                    icomp, i, (i_start+shift):(i_end+shift)]

                distance = window.get_epicentral_distance()
                max_obs = np.abs(data_cut).max() * 2
                ts = np.linspace(
                    0, len(data_cut)/output.sampling_hz, len(data_cut))
                ax1.plot(ts, data_cut/max_obs+distance, color='black')
                ax1.plot(ts, u_cut/max_obs+distance, color='red')
                ax2.plot(ts, data_cut/max_obs+distance, color='black')
                ax2.plot(ts, u_gcmt_cut/max_obs+distance, color='green')

                # plt.show()

        ax1.title.set_text('Ours')
        ax2.title.set_text('GCMT')
        ax1.set(
            ylabel='Distance (deg)',
            xlabel='Time (s)')
        ax2.set(xlabel='Time (s)')
        plt.suptitle(event_id)

        plt.savefig(filename, bbox_inches='tight')

    @staticmethod
    def _misfit(obs, syn):
        corr = np.corrcoef(obs, syn)[0, 1]
        amp_ratio = (obs.max() - obs.min()) / (syn.max() - syn.min())
        return 0.5*(1 - corr) + np.abs(np.log(0.333)*np.log(amp_ratio))


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    n_cores = comm.Get_size()
    rank = comm.Get_rank()

    model = SeismicModel.prem()
    tlen = 3276.8
    nspc = 1024
    sampling_hz = 20
    freq = 0.005
    freq2 = 0.167
    duration_min = 1.
    duration_max = 15.
    duration_inc = 1.
    amp = 3.
    amp_inc = .2
    distance_min = 10.
    distance_max = 90.
    dir_syn = '.'

    logfile = open('log_{}'.format(rank), 'w', buffering=1)

    for sac_files in sac_files_iterator(
        '/mnt/doremi/anpan/inversion/MTZ_JAPAN/DATA/20*/*T',
        comm, log=logfile):
        #'/mnt/doremi/anpan/inversion/MTZ_JAPAN/DATA/20*/*T'
        #'/work/anselme/DATA/CENTRAL_AMERICA/2005*/*T'

        logfile.write('{} num sacs = {}\n'.format(rank, len(sac_files)))

        if rank == 0:
            logfile.write('{} reading dataset\n'.format(rank))
            dataset = Dataset.dataset_from_sac(sac_files, headonly=False)

            logfile.write('{} computing time windows\n'.format(rank))
            windows_S = WindowMaker.windows_from_dataset(
                dataset, 'prem', ['s', 'S', 'Sdiff'],
                [Component.T], t_before=10., t_after=20.)
            windows_P = WindowMaker.windows_from_dataset(
                dataset, 'prem', ['p', 'P', 'Pdiff'],
                [Component.Z], t_before=10., t_after=20.)
            windows = windows_S #+ windows_P
            windows = [
                window for window in windows
                if (
                    window.get_epicentral_distance() >= distance_min
                    and window.get_epicentral_distance() <= distance_max)]
        else:
            dataset = None
            windows = None

        durations = np.linspace(
                duration_min, duration_max,
                int((duration_max - duration_min)/duration_inc)+1)
        amplitudes = np.linspace(1., amp, int((amp-1)/amp_inc)+1)
        amplitudes = np.concatenate((1./amplitudes[:0:-1], amplitudes))

        stfgrid = STFGridSearch(
            dataset, model, tlen, nspc, sampling_hz, freq, freq2, windows,
            durations, amplitudes)

        logfile.write('{} computing synthetics\n'.format(rank))
        misfit_dict = stfgrid.compute_parallel(
            comm, mode=2, dir=dir_syn, verbose=2, log=logfile)
        
        if rank == 0:
            logfile.write('{} saving misfits\n'.format(rank))
            best_params_dict = stfgrid.get_best_parameters(misfit_dict)

            catalog_name = 'stf_catalog.txt'
            stfgrid.save_catalog(catalog_name, best_params_dict)

            for event_id in misfit_dict.keys():
                filename = '{}.pdf'.format(event_id)
                logfile.write(
                    '{} saving figure to {}\n'.format(rank, filename))
                stfgrid.savefig(best_params_dict, event_id, filename)

    logfile.write('{} Done!\n'.format(rank))
    logfile.close()
    
