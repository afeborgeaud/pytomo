from pydsm.windows import Windows
from pydsm.dataset import Dataset
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from stream import read_sac

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
            freq=1.):
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

        self.filter_traces(freq=freq)
        # TODO resample to same sampling

    def compute(self):
        """Compute source wavelets for all events.

        Return:
            wavelet_dict (dict): dictionary of wavelets for each event_id
        """
        wavelet_dict = dict()
        windows = self.compute_windows()
        self.windows = windows

        wavelet_dict = self.initialize(windows)

        wavelet_dict = self.run_one_iteration(windows, wavelet_dict)

        wavelet_dict = self.run_one_iteration(windows, wavelet_dict)

        masks = self.remove_bad_windows(windows, wavelet_dict, self.min_cc)

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
        Return: windows (list): list of pydsm.Windows
        """
        # TODO speed up by grouping events
        windows = []
        for trace in self.traces:
            winmaker = Windows.windows_from_obspy_traces(
                trace, self.modelname, self.phasenames)
            # keep only the first arrival
            window = winmaker.get_windows(self.t_before, self.t_after)[0]
            windows.append(window)
        return windows

    def filter_traces(self, freq=1., zerophase=True):
        for i, trace in enumerate(self.traces):
            self.traces[i] = trace.filter(
                'lowpass', freq=freq, zerophase=zerophase)

    def initialize(self, windows):
        npts = int((windows[0][1] - windows[0][0]) * self.sampling)
        wavelet_dict = dict()
        for trace in self.traces:
            wavelet_dict[trace.stats.sac.kevnm] = np.zeros(
                npts, dtype=np.float32)

        for window, trace in zip(windows, self.traces):
            if np.isnan(window[0]) or np.isnan(window[1]):
                continue
            start = int(window[0] * self.sampling)
            end = int(window[1] * self.sampling)

            waveform_cut = np.array(trace.data[start:end])
            waveform_cut /= np.max(np.abs(waveform_cut))

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
            if np.isnan(window[0]) or np.isnan(window[1]):
                continue
            if not mask:
                continue
            start = int((window[0]-buffer) * self.sampling)
            end = int((window[1]+buffer) * self.sampling)
            
            wavelet0 = wavelet_dict[trace.stats.sac.kevnm]

            waveform_cut_buffer = np.array(trace.data[start:end])
            best_shift, polarity = self.find_best_shift(
                waveform_cut_buffer, wavelet0)
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
            if np.isnan(window[0]) or np.isnan(window[1]) or not mask:
                traces_aligned.append(trace)
                continue

            start = int((window[0]-buffer) * self.sampling)
            end = int((window[1]+buffer) * self.sampling)

            wavelet = wavelet_dict[trace.stats.sac.kevnm]

            waveform_cut_buffer = np.array(trace.data[start:end])
            best_shift, polarity = self.find_best_shift(
                waveform_cut_buffer, wavelet)
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
            if np.isnan(window[0]) or np.isnan(window[1]) or not mask:
                continue
            t = trace.stats.starttime
            b = trace.stats.sac.b
            traces_cut.append(trace.slice(t+window[0]+b, t+window[1]+b))
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
            if np.isnan(window[0]) or np.isnan(window[1]):
                masks[i] = False
                continue
            start = int(window[0] * self.sampling)
            end = int(window[1] * self.sampling)

            wavelet = wavelet_dict[trace.stats.sac.kevnm]
            waveform_cut = trace.data[start:end]
            corr = np.corrcoef(waveform_cut, wavelet)[0,1]
            if corr < min_cc:
                masks[i] = False 
        
        return masks

    def find_best_shift(self, y, y_template):
        n = len(y_template)
        n_shift = len(y) - n
        corrs = np.zeros(n_shift)
        for i in range(n_shift):
            y_shift = y[i:i+n]
            corrs[i] = np.corrcoef(y_shift, y_template)[0,1]
        if corrs.max() >= -corrs.min():
            best_shift = np.argmax(corrs)
            polarity = 1.
        else:
            best_shift = np.argmin(corrs)
            polarity = -1.
        if not self.shift_polarity:
            polarity = 1.
        return best_shift, polarity

    def process_causal_wavelet(self, wavelet_dict, eps=0.01):
        stf_dict = dict(wavelet_dict)
        for event_id in wavelet_dict.keys():
            wavelet = wavelet_dict[event_id]
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

    def plot(self):
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
                    ax1.plot(trace.times(), waveform_norm+dist, 'b-')
            ts = np.arange(len(self.wavelet_dict[event_id])) / self.sampling
            ax1.plot(ts, self.wavelet_dict[event_id]+dist_min-1, 'r-')
            ts = np.arange(len(self.stf_dict[event_id])) / self.sampling
            ax1.plot(ts, self.stf_dict[event_id]+dist_min-2, 'r-')
            for trace in traces_original:
                if trace.stats.sac.kevnm == event_id:
                    dist = trace.stats.sac.gcarc
                    waveform_norm = (trace.data 
                        / np.abs(trace.data).max())
                    ax2.plot(trace.times(), waveform_norm+dist, 'b-')
            ax1.set(
                xlabel='Time (s)',
                ylabel='Distance (deg)')
            ax2.set(
                xlabel='Time (s)')
            plt.show()

if __name__ == '__main__':
    sac_names = ('/work/anselme/CA_ANEL_NEW/VERTICAL/200503211223A/*Z')
    traces = read_sac(sac_names)

    iterstack = IterStack(
        traces, freq=1., shift_polarity=True, min_cc=0.6)
    iterstack.compute()
    iterstack.plot()
