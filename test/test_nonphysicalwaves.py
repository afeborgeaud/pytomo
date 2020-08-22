from pydsm.seismicmodel import SeismicModel
from pydsm.station import Station
from pydsm.event import Event
from pydsm.utils.cmtcatalog import read_catalog
from pydsm.dsm import PyDSMInput, compute
from pydsm.component import Component
import numpy as np
import matplotlib.pyplot as plt

def get_output(tlen=1638.4, nspc=64, sampling_hz=20, mode=0):
    catalog = read_catalog()
    event = Event.event_from_catalog(
        catalog, '200707211534A')
    stations = [
        Station(
            '{:03d}'.format(i), 'DSM', event.latitude, event.longitude+i)
        for i in range(12,36)]
    
    model = SeismicModel.ak135()
    pydsm_input = PyDSMInput.input_from_arrays(
        event, stations, model, tlen, nspc, sampling_hz)
    pydsm_output = compute(pydsm_input, mode=mode)
    pydsm_output.to_time_domain()

    return pydsm_output

if __name__ == '__main__':
    output_sh = get_output(nspc=256, mode=2)
    output_shpsv = get_output(nspc=256, mode=0)

    output_sh.filter(0.005, 0.08, type='bandpass')
    output_shpsv.filter(0.005, 0.08, type='bandpass')

    fig, ax = plt.subplots(1)
    _, ax = output_sh.plot_component(
            Component.T, ax=ax, align_zero=True, color='black')
    _, ax = output_shpsv.plot_component(
            Component.T, ax=ax, align_zero=True, color='red')       
    ax.set_xlim(0, 1200)
    plt.show()