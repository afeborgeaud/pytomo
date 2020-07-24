from pydsm.windows import Windows
from pydsm.dataset import Dataset
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

class IterStack:
    """Iterative stacking for source wavelet
    following Kennett & Rowlingson (2014)

    Args:
        traces (obspy.traces): waveform traces
    """

    def __init__(
            self, traces,
            modelname='prem', phasename='P',
            t_before=10, t_after=30., min_cc=0.6):
        self.ds = ds
        self.modelname = modelname
        self.phasename = phasename
        self.t_before = t_before
        self.t_after = t_after
        self.min_cc = min_cc
        # TODO resample to same sampling

    def compute(self):
        """Compute source wavelets for all events.

        Return:
            wavelet_dict (dict): dictionary of wavelets for each event_id
        """
        wavelet_dict = dict()
        for i in range(len(self.ds.events)):
            start, end = self.ds.get_bounds_from_event_index(i)
            winmaker = Windows(
                self.ds.events[i], self.ds.stations[start:end],
                self.modelname, [self.phasename,])
            winmaker.compute()
            windows = winmaker.get_windows(self.t_before, self.t_after)

            wavelet0, waveforms_cut0 = self.initialize(
                self.ds.data[start:end], windows, self.ds.sampling)

            wavelet1, waveforms_cut1 = self.run_one_iteration(
                wavelet0, self.ds.data[start:end],
                windows, self.ds.sampling)

            masks = self.remove_bad_windows(
                wavelet1, self.ds.data[start:end],
                windows, self.ds.sampling, self.min_cc)

            wavelet2, waveforms_cut2 = self.run_one_iteration(
                wavelet1, self.ds.data[start:end],
                windows, self.ds.sampling, masks=masks)

            wavelet_causal = self.process_causal_wavelet(wavelet2)
            wavelet_dict[self.ds.events[i].event_id] = wavelet_causal

            fig1, ax1 = self.record_section(wavelet_causal, waveforms_cut2)
            fig2, ax2 = self.record_section(wavelet0, waveforms_cut0)
            plt.show()
        return wavelet_dict

    def initialize(self, waveforms, windows, sampling):
        npts = int((windows[0][1] - windows[0][0]) * sampling)
        wavelet = np.zeros(npts, dtype=np.float32)
        waveforms_cut = []
        for window, waveform in zip(windows, waveforms):
            if len(waveform) != len(waveforms[0]):
                continue
            if np.isnan(window[0]) or np.isnan(window[1]):
                continue
            start = int(window[0] * sampling)
            end = int(window[1] * sampling)

            waveform_cut = np.array(waveform[start:end])
            waveform_cut /= np.max(np.abs(waveform_cut))
            waveforms_cut.append(waveform_cut)

            wavelet += waveform_cut
        wavelet /= np.max(np.abs(wavelet))
        return wavelet, waveforms_cut

    def run_one_iteration(
            self, wavelet0, waveforms, windows, sampling, buffer=10,
            masks=None):
        npts = int((windows[0][1] - windows[0][0]) * sampling)
        wavelet = np.zeros(npts, dtype=np.float32)
        waveforms_cut = []
        if masks is None:
            masks = np.full(len(windows), True, dtype=np.bool)
        for window, waveform, mask in zip(windows, waveforms, masks):
            if len(waveform) != len(waveforms[0]):
                continue
            if np.isnan(window[0]) or np.isnan(window[1]):
                continue
            if not mask:
                continue
            start = int((window[0]-buffer) * sampling)
            end = int((window[1]+buffer) * sampling)

            waveform_cut_buffer = np.array(waveform[start:end])
            best_shift = self.find_best_shift(waveform_cut_buffer, wavelet0)
            waveform_cut = waveform_cut_buffer[best_shift:best_shift+npts]
            waveform_cut /= np.max(np.abs(waveform_cut))
            waveforms_cut.append(waveform_cut)

            wavelet += waveform_cut
        wavelet /= np.max(np.abs(wavelet))
        return wavelet, waveforms_cut

    def remove_bad_windows(
        self, wavelet, waveforms, windows, sampling, min_cc):
        masks = np.full(len(windows), True, dtype=np.bool)
        for i, (window, waveform) in enumerate(zip(windows, waveforms)):
            if len(waveform) != len(waveforms[0]):
                masks[i] = False
                continue
            if np.isnan(window[0]) or np.isnan(window[1]):
                masks[i] = False
                continue
            start = int(window[0] * sampling)
            end = int(window[1] * sampling)

            waveform_cut = waveform[start:end]
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
        best_shift = np.argmax(corrs)
        return best_shift

    def process_causal_wavelet(self, wavelet, eps=0.01):
        # integrate to get source-time function
        stf = np.cumsum(wavelet) / self.ds.sampling

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
        integ = np.abs(stf_causal.sum()) / self.ds.sampling
        stf_causal /= integ

        return stf_causal
    
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

    def record_section(self, wavelet, waveforms):
        fig, ax = plt.subplots(1,1)
        xs = list(range(len(waveforms[0])))
        ax.plot(xs[:len(wavelet)], wavelet, 'r-')
        for i, waveform in enumerate(waveforms):
            ax.plot(xs, waveform+i+1, 'b-')
        ax.set(
            xlabel='Time (s)',
            ylabel='Wave. index')
        return fig, ax

if __name__ == '__main__':
    sac_names = ('/work/anselme/CA_ANEL_NEW/VERTICAL/syntheticPREM_Q165/'
        + 'filtered_stf_8-200s/200503211223A/*Z')
    full_path = os.path.expanduser(sac_names)
    sac_files = list(glob.iglob(full_path))
    ds = Dataset.dataset_from_sac(sac_files, headonly=False)

    iterstack = IterStack(ds)
    wavelet_dict = iterstack.compute()




