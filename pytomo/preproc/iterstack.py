from dsmpy.window import Window
from dsmpy.windowmaker import WindowMaker
from dsmpy.dataset import Dataset
from .stream import read_sac
import numpy as np
import os
import glob
import sys
import time
import matplotlib.pyplot as plt
import warnings

class InputFile:
    """Input file for IterStack.

    Args:
        input_file (str): path of IterStack input file
    """
    def __init__(self, input_file):
        self.input_file = input_file
    
    def read(self):
        params = dict()
        params['verbose'] = 0
        params['modelname'] = 'prem'
        params['phasenames'] = ['p', 'P', 'Pdiff']
        params['min_cc'] = 0.6
        params['freq'] = 0.005
        params['freq2'] = 1.
        params['shift_polarity'] = True
        params['t_before'] = 10
        params['t_after'] = 30
        with open(self.input_file, 'r') as f:
            for line in f:
                if line.strip().startswith('#'):
                    continue
                key, value = self._parse_line(line)
                if key is not None:
                    params[key] = value
        return params

    def _parse_line(self, line):
        key, value = line.strip().split()[:2]
        if key == 'sac_files':
            full_path = os.path.expanduser(value.strip())
            value_parsed = full_path
        elif key == 'freq':
            value_parsed = float(value)
        elif key == 'freq2':
            value_parsed = float(value)
        elif key == 'min_cc':
            value_parsed = float(value)
        elif key == 't_before':
            value_parsed = float(value)
        elif key == 't_after':
            value_parsed = float(value)
        elif key == 'phasenames':
            values = line.strip().split()[1:]
            value_parsed = [name.strip() for name in values]
        elif key == 'modelname':
            value_parsed = value.strip().lower()
        elif key == 'shift_polarity':
            value_parsed = bool(value)
        elif key == 'out_dir':
            full_path = os.path.expanduser(value.strip())
            value_parsed = full_path
        elif key == 'verbose':
            value_parsed = int(value)
        else:
            print('Warning: key {} undefined. Ignoring.'.format(key))
            return None, None
        return key, value_parsed

def find_best_shift(
        y, y_template, shift_polarity=False, skip_freq=1):
    """Compute the index shift to maximize the correlation
     between y[shift:shift+n] and y_template,
     where n=len(y_template)

    Args:
        y (np.ndarray): must have len(y) >= len(y_template)
        y_template (np.ndarray):
        shift_polarity: allows to switch polarity (default is False)
        skip_freq: skip points to reduce computation
            time (default is 1, i.e. no skip)

    Returns:
        int: best shift
        int: polarity (1 or -1)
    """
    if not np.any(y) or not np.any(y_template):
        warnings.warn('y or y_template is 0. Returning 0 shift')
        return 0, 1.
    n = len(y_template)
    n_shift = len(y) - n
    assert n_shift % skip_freq == 0
    n_shift = int(n_shift / skip_freq)
    corrs = np.zeros(n_shift)
    for i in range(n_shift):
        y_shift = y[i*skip_freq:i*skip_freq+n]
        corrs[i] = np.corrcoef(y_shift, y_template)[0, 1]
    if not shift_polarity:
        best_shift = np.argmax(corrs) * skip_freq
        polarity = 1.
    else:
        if corrs.max() < -corrs.min():
            best_shift = np.argmin(corrs) * skip_freq
            polarity = -1.
    return best_shift, polarity

class IterStack:
    """Iterative stacking for source wavelet
    following Kennett & Rowlingson (2014)

    Args:
        traces (obspy.traces): waveform traces
    """

    def __init__(
            self, traces,
            modelname='prem', phasenames=['p', 'P', 'Pdiff'],
            t_before=10, t_after=30., min_cc=0.6,
            sampling=20, shift_polarity=True,
            freq=0.005, freq2=1., verbose=0):
        assert freq2 > freq

        self.traces = []
        for i in range(len(traces)):
            self.traces.append(traces[i].copy())
        self.modelname = modelname
        self.phasenames = phasenames
        self.t_before = t_before
        self.t_after = t_after
        self.min_cc = min_cc
        self.sampling = sampling
        self.shift_polarity = shift_polarity
        self.verbose = verbose

        self.filter_traces(freq, freq2)
        # TODO resample to same sampling

    def compute(self):
        """Compute source wavelets for all events.

        Return:
            wavelet_dict (dict): dictionary of wavelets for each event_id
        """
        wavelet_dict = dict()
        self.count_dict = dict()
        windows = self.compute_windows()
        self.windows = windows
        if self.verbose > 0:
            print(
                'Number of time windows before selection: {}'
                .format(len(windows)))

        wavelet_dict = self.initialize(windows)

        wavelet_dict = self.run_one_iteration(windows, wavelet_dict)

        wavelet_dict = self.run_one_iteration(windows, wavelet_dict)

        masks = self.remove_bad_windows(windows, wavelet_dict, self.min_cc)
        if self.verbose > 0:
            print(
                'Number of time windows after selection: {}'
                .format((masks==True).sum()))

        for i, window in enumerate(self.windows):
            if masks[i]:
                if window.event.event_id not in self.count_dict:
                    self.count_dict[window.event.event_id] = 1
                else:
                    self.count_dict[window.event.event_id] += 1

        wavelet_dict = self.run_one_iteration(
            windows, wavelet_dict, masks=masks)

        stf_dict = self.process_causal_wavelet(wavelet_dict)

        self.wavelet_dict = wavelet_dict
        self.stf_dict = stf_dict
        self.masks = masks

    def get_aligned_traces(self, cut=True, select=True):
        aligned_traces = self.align_traces(self.windows, self.wavelet_dict)
        masks = self.masks if select else None
        if cut:
            aligned_traces = self.cut_traces(
                self.windows, aligned_traces, masks)
        return aligned_traces

    def get_original_traces(self, cut=True, select=True):
        masks = self.masks if select else None
        if cut:
            return self.cut_traces(
                self.windows, self.traces, masks)
        else:
            return self.traces

    def compute_windows(self):
        """
        Return: windows (list): list of pydsm.window.Window
        """
        # TODO speed up by grouping events
        windows = []
        for trace in self.traces:
            # keep only the first arrival
            window = WindowMaker.windows_from_obspy_traces(
                trace, self.modelname, self.phasenames,
                self.t_before, self.t_after)[0]
            windows.append(window)
        return windows

    def filter_traces(self, freqmin=0.005, freqmax=1., zerophase=True):
        for i, trace in enumerate(self.traces):
            self.traces[i] = trace.filter(
                'bandpass', freqmin=freqmin, freqmax=freqmax,
                zerophase=zerophase)

    def initialize(self, windows):
        window_arr = windows[0].to_array()
        npts = int((window_arr[1] - window_arr[0]) * self.sampling)
        wavelet_dict = dict()
        for trace in self.traces:
            wavelet_dict[trace.stats.sac.kevnm] = np.zeros(
                npts, dtype=np.float32)

        for window, trace in zip(windows, self.traces):
            window_arr = window.to_array()
            if np.isnan(window_arr[0]) or np.isnan(window_arr[1]):
                continue
            start = int(window_arr[0] * self.sampling)
            end = int(window_arr[1] * self.sampling)

            waveform_cut = np.array(trace.data[start:end])
            waveform_cut /= np.max(np.abs(waveform_cut))
            
            try:
                wavelet_dict[trace.stats.sac.kevnm] += waveform_cut
            except:
                n1 = len(waveform_cut)
                n2 = len(wavelet_dict[trace.stats.sac.kevnm])
                if n1 < n2:
                    waveform_cut = np.pad(
                        waveform_cut, (0,n2-n1), mod='constant',
                        constant_values=(0,0))
                else:
                    waveform_cut = waveform_cut[:n2]
                wavelet_dict[trace.stats.sac.kevnm] += waveform_cut

        for event_id in wavelet_dict.keys():
            wavelet_dict[event_id] /= np.max(np.abs(wavelet_dict[event_id]))
        return wavelet_dict

    def run_one_iteration(
            self, windows, wavelet_dict, buffer=10,
            masks=None):
        wavelet_dict_ = dict(wavelet_dict)
        if masks is None:
            masks = np.full(len(windows), True, dtype=np.bool)
        for window, trace, mask in zip(windows, self.traces, masks):
            window_arr = window.to_array()
            if np.isnan(window_arr[0]) or np.isnan(window_arr[1]):
                continue
            if not mask:
                continue
            start = int((window_arr[0]-buffer) * self.sampling)
            end = int((window_arr[1]+buffer) * self.sampling)
            
            wavelet0 = wavelet_dict[trace.stats.sac.kevnm]

            waveform_cut_buffer = np.array(trace.data[start:end])
            best_shift, polarity = find_best_shift(
                waveform_cut_buffer, wavelet0, self.shift_polarity)
            waveform_cut = waveform_cut_buffer[
                best_shift:best_shift+len(wavelet0)]
            waveform_cut /= np.max(np.abs(waveform_cut)) * polarity

            wavelet_dict_[trace.stats.sac.kevnm] += waveform_cut
        for event_id in wavelet_dict_.keys():
            wavelet_dict_[event_id] /= np.max(np.abs(wavelet_dict_[event_id]))
        return wavelet_dict_
    
    def align_traces(
            self, windows, wavelet_dict, buffer=10.,
            masks=None):
        if masks is None:
            masks = np.full(len(windows), True, dtype=np.bool)
        traces_aligned = []
        for window, trace, mask in zip(windows, self.traces, masks):
            window_arr = window.to_array()
            if np.isnan(window_arr[0]) or np.isnan(window_arr[1]) or not mask:
                traces_aligned.append(trace)
                continue

            start = int((window_arr[0]-buffer) * self.sampling)
            end = int((window_arr[1]+buffer) * self.sampling)

            wavelet = wavelet_dict[trace.stats.sac.kevnm]

            waveform_cut_buffer = np.array(trace.data[start:end])
            best_shift, polarity = find_best_shift(
                waveform_cut_buffer, wavelet, self.shift_polarity)
            best_shift = best_shift/self.sampling - buffer
            
            trace_aligned = IterStack.shift_trace(trace, best_shift)
            trace_aligned.data *= polarity
            traces_aligned.append(trace_aligned)
        return traces_aligned

    def cut_traces(self, windows, traces, masks=None):
        traces_cut = []
        if masks is None:
            masks = np.full(len(windows), True, dtype=np.bool)
        for window, trace, mask in zip(windows, traces, masks):
            window_arr = window.to_array()
            if np.isnan(window_arr[0]) or np.isnan(window_arr[1]) or not mask:
                continue
            t = trace.stats.starttime
            b = trace.stats.sac.b
            traces_cut.append(
                trace.slice(t+window_arr[0]+b, t+window_arr[1]+b))
        return traces_cut
    
    @staticmethod
    def shift_trace(trace, shift):
        trace_shift = trace.copy()
        trace_shift.stats.sac.b += shift
        trace_shift.stats.sac.e += shift
        return trace_shift

    def remove_bad_windows(
        self, windows, wavelet_dict, min_cc):
        masks = np.full(len(windows), True, dtype=np.bool)
        for i, (window, trace) in enumerate(zip(windows, self.traces)):
            window_arr = window.to_array()
            if np.isnan(window_arr[0]) or np.isnan(window_arr[1]):
                masks[i] = False
                continue
            start = int(window_arr[0] * self.sampling)
            end = int(window_arr[1] * self.sampling)

            wavelet = wavelet_dict[trace.stats.sac.kevnm]
            waveform_cut = trace.data[start:end]
            
            n1 = len(waveform_cut)
            n2 = len(wavelet)
            if n1 < n2:
                waveform_cut = np.pad(
                    waveform_cut, (0,n2-n1), mod='constant',
                    constant_values=(0,0))
            else:
                waveform_cut = waveform_cut[:n2]

            corr = np.corrcoef(waveform_cut, wavelet)[0,1]
            if corr < min_cc:
                masks[i] = False 
        
        return masks

    def process_causal_wavelet(self, wavelet_dict, eps=0.01):
        stf_dict = dict(wavelet_dict)
        for event_id in wavelet_dict.keys():
            wavelet = wavelet_dict[event_id]
            if wavelet.max() < -wavelet.min():
                wavelet *= -1
            # integrate to get source-time function
            stf = np.cumsum(wavelet) / self.sampling

            i_peak = np.argmax(stf)

            i = i_peak
            while ((i > 0)
                and (np.abs(stf[i]) > eps * np.abs(stf[i_peak]))):
                i -= 1
            i0 = i

            i = i_peak
            while ((i < len(stf))
                and (np.abs(stf[i]) > eps * np.abs(stf[i_peak]))):
                i += 1
            i1 = i + 1

            stf_causal = stf[i0:i1]

            # taper
            stf_causal = IterStack.planck_taper(stf_causal)
            # normalize integral
            integ = np.abs(stf_causal.sum()) / self.sampling
            stf_causal /= integ

            stf_dict[event_id] = stf_causal
        return stf_dict
    
    @staticmethod
    def planck_taper(wavelet, f=0.1):
        n = len(wavelet)
        i0 = int(n * f)
        tap = np.ones(n, dtype=np.float32)
        tap[0] = 0.
        tap[n-1] = 0
        for i in range(1, i0):
            tap[i] = 1 / (1 + np.exp(f*n/i - f*n/(f*n-i)))
            tap[n-i-1] = tap[i]
        return wavelet * tap

    def save_figure(self, out_dir, **kwargs):
        traces_aligned = self.get_aligned_traces()
        traces_original = self.get_original_traces()

        for event_id in self.wavelet_dict.keys():
            fig, (ax1, ax2) = plt.subplots(
                1, 2, sharey=True, sharex=True)
            dist_min = 360.
            for trace in traces_aligned:
                if trace.stats.sac.kevnm == event_id:
                    dist = trace.stats.sac.gcarc
                    if dist < dist_min:
                        dist_min = dist
                    waveform_norm = (trace.data 
                        / np.abs(trace.data).max())
                    ax1.plot(
                        trace.times(), waveform_norm+dist, 'b-', **kwargs)
            ts = np.arange(len(self.wavelet_dict[event_id])) / self.sampling
            ax1.plot(
                ts, self.wavelet_dict[event_id]+dist_min-1, 'r-', **kwargs)
            ts = np.arange(len(self.stf_dict[event_id])) / self.sampling
            ax1.plot(
                ts, self.stf_dict[event_id]+dist_min-2, 'r-', **kwargs)
            for trace in traces_original:
                if trace.stats.sac.kevnm == event_id:
                    dist = trace.stats.sac.gcarc
                    waveform_norm = (trace.data 
                        / np.abs(trace.data).max())
                    ax2.plot(
                        trace.times(), waveform_norm+dist, 'b-', **kwargs)
            ax1.set(
                xlabel='Time (s)',
                ylabel='Distance (deg)')
            ax2.set(
                xlabel='Time (s)')
            
            out_name = out_dir + '/' + event_id + '.pdf'
            plt.savefig(out_name)
            plt.close(fig)
    
    def save_stf_catalog(self, out_dir):
        '''Save the empirical source time functions to files. One file
        per event, named event_id.stf.
        Args:
            out_dir (str): output directory
        '''
        for event_id in self.stf_dict.keys():
            file_name = out_dir + '/' + event_id + '.stf'
            ts = np.linspace(
                0,
                len(self.stf_dict[event_id])/self.sampling,
                len(self.stf_dict[event_id]))
            stf = np.vstack((ts, self.stf_dict[event_id])).T
            np.savetxt(
                file_name, stf)
        catalog_info_name = out_dir + '/catalog_info.txt'
        with open(catalog_info_name, 'w') as f:
            f.write('id duration num_waveforms\n')
            for event_id in self.stf_dict.keys():
                try:
                    f.write(
                        '{} {} {}\n'.format(
                            event_id,
                            len(self.stf_dict[event_id])/self.sampling,
                            self.count_dict[event_id]))
                except:
                    print('Problem with key {}'.format(event_id))

if __name__ == '__main__':
    # sac_path = '/work/anselme/CA_ANEL_NEW/VERTICAL/200503211223A/*Z'
    # out_dir = 'figures'
    # sac_names = (sac_path)
    # traces = read_sac(sac_names)

    input_file = InputFile(sys.argv[1])
    params = input_file.read()

    sac_files = params['sac_files']
    min_cc = params['min_cc']
    freq = params['freq']
    freq2 = params['freq2']
    t_before = params['t_before']
    t_after = params['t_after']
    phasenames = params['phasenames']
    modelname = params['modelname']
    shift_polarity = params['shift_polarity']
    out_dir = params['out_dir']
    verbose = params['verbose']

    start_time = time.time_ns()
    if verbose > 0:
        print('Reading sac files')
    traces = read_sac(sac_files)

    iterstack = IterStack(
        traces, modelname, phasenames, t_before, t_after,
        min_cc, freq=freq, freq2=freq2, shift_polarity=shift_polarity,
        verbose=verbose)

    if verbose > 0:
        print('Start computing stf')
    iterstack.compute()
    
    end_time = time.time_ns()
    if verbose > 0:
        print('STF computed in {} s'.format((end_time-start_time)*1e-9))

    iterstack.save_figure(out_dir)
    iterstack.save_stf_catalog(out_dir)
    
