import numpy as np
from .stream import read_sac
from .iterstack import IterStack
import os
import glob
import matplotlib.pyplot as plt

class Deconvolver:
    """Perform deconvolution.

    Args:
        traces (obspy.traces): waveform traces
        wavelet_dict (dict): dictionary of wavelets for each event_id
    """

    def __init__(self, traces, wavelet_dict):
        self.traces = []
        for i in range(len(traces)):
            trace_filt = (traces[i].copy()
                          .filter('lowpass', freq=2., zerophase=False))
            self.traces.append(trace_filt)
        self.wavelet_dict = wavelet_dict
    
    def run(self):
        for i in range(len(traces)):
            wavelet = self.wavelet_dict[self.traces[i].stats.sac.kevnm]

            u_deconv = self.deconvolve(self.traces[i].data, wavelet)
            # plt.plot(self.ds.data[j], label='raw')
            # plt.plot(u_deconv, label='deconv')
            # plt.legend()
            # plt.show()
            self.traces[i].data = u_deconv

    def deconvolve(self, waveform, wavelet, level=0.005):
        """Deconvolve a single waveform.
        Args:
            waveform (ndarray): waveform
            wavelet (ndarray): wavelet
            level (float): water level
        """
        n = len(waveform)
        ft = 20 * 10.
        f = ft / n
        waveform_tap = IterStack.planck_taper(waveform, f=f)
        n_wavelet = len(wavelet)
        wavelet_pad = np.pad(wavelet, pad_width=(0, n-len(wavelet)),
                            mode='constant', constant_values=0)
        u_fft = np.fft.fft(waveform_tap)
        wavelet_fft = np.fft.fft(wavelet_pad)
        wavelet_max = np.abs(wavelet_fft).max()
        wavelet_fft /= -wavelet_max

        # plt.plot(np.abs(wavelet_fft), label='wavelet')
        # plt.plot(np.abs(u_fft)/np.abs(u_fft).max(), label='u')
        # plt.plot(wavelet_pad)
        # plt.legend()
        # plt.show()

        u_deconv_fft = np.divide(u_fft, wavelet_fft,
            where=(np.abs(wavelet_fft) > np.abs(wavelet_fft).max() * level))
        # u_deconv_fft = u_fft / (wavelet_fft + np.abs(wavelet_fft).max() * level)
        # u_deconv_fft = u_fft / wavelet_fft
        u_deconv = np.real(np.fft.ifft(u_deconv_fft))
        return u_deconv

if __name__ == '__main__':
    sac_names = ('/work/anselme/CA_ANEL_NEW/VERTICAL/200503211223A/*Z')
    traces = read_sac(sac_names)

    iterstack = IterStack(
        traces, freq=5., shift_polarity=True, min_cc=0.6)
    iterstack.compute()
    iterstack.plot()

    deconvolver = Deconvolver(traces, iterstack.stf_dict)
    deconvolver.run()

    iterstack = IterStack(
        deconvolver.traces, freq=.5, shift_polarity=True, min_cc=0.6)
    iterstack.compute()
    iterstack.plot()