from pydsm.seismicmodel import SeismicModel
from pydsm.station import Station
from pydsm.event import Event
from pydsm.utils.cmtcatalog import read_catalog
from pydsm.dsm import PyDSMInput, compute
from pydsm.component import Component
import numpy as np
import matplotlib.pyplot as plt

def get_output(tlen=1638.4, nspc=64, sampling_hz=20):
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
    pydsm_output = compute(pydsm_input, mode=0)
    pydsm_output.to_time_domain()

    return pydsm_output

if __name__ == '__main__':
    # tlens = [1638.4, 1400., 1200.]
    tlens = [1638.4, 1280.]
    nspcs = [64, 50]
    outputs = [get_output(tlen=tlen, nspc=nspc*2)
               for tlen,nspc in zip(tlens,nspcs)]

    fig, ax = plt.subplots(1)
    cycler = plt.rcParams['axes.prop_cycle']
    for tlen, output, sty in zip(tlens, outputs, cycler[:len(tlens)]):
        _, ax = output.plot_component(
            Component.T, ax=ax, align_zero=True, **sty)
    ax.set_xlim(0, tlens[-1])
    plt.show()