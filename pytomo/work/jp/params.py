from pydsm.modelparameters import ModelParameters, ParameterType
from pydsm.seismicmodel import SeismicModel
from pydsm.station import Station
from pydsm.event import Event
from pydsm.utils.cmtcatalog import read_catalog
from pydsm.dataset import Dataset
from pydsm.dsm import PyDSMInput, compute
import numpy as np

def get_model(
        n_upper_mantle=20, n_mtz=10, n_lower_mantle=12,
        types=[ParameterType.VSH], verbose=0):
    '''Boxcar mesh using ak135 as reference model for the structure of
        the upper mantle and transition zone down to 1000 km depth.
    '''
    ak135 = SeismicModel.ak135()
    # model parameters
    depth_moho = 6371. - 6336.6
    depth_410 = 410.
    depth_660 = 660.
    depth_max = 1000.
    rs_upper_mantle = np.linspace(depth_410, depth_moho, n_upper_mantle+1)
    rs_mtz = np.linspace(depth_660, depth_410, n_mtz, endpoint=False)
    rs_lower_mantle = np.linspace(
        depth_max, depth_660, n_lower_mantle, endpoint=False)
    radii = 6371. - np.round(
        np.hstack((rs_lower_mantle, rs_mtz, rs_upper_mantle)), 4)
    if verbose > 1:
        print('dr_um={}, dr_mtz={}, dr_lm={}'.format(
            rs_upper_mantle[1] - rs_upper_mantle[0],
            rs_mtz[1] - rs_mtz[0],
            rs_lower_mantle[1] - rs_lower_mantle[0]))

    model_params = ModelParameters(types, radii, mesh_type='boxcar')
    # mesh
    # model, mesh = ak135.boxcar_mesh(model_params)
    
    return ak135, model_params

def get_dataset(tlen=1638.4, nspc=64, sampling_hz=20, mode=0):
    catalog = read_catalog()
    event = Event.event_from_catalog(
        catalog, '200707211534A')
    events = [event]
    stations = [
        Station(
            '{:03d}'.format(i), 'DSM', event.latitude, event.longitude+i)
        for i in range(12,36)]
    dataset = Dataset.dataset_from_arrays(
        events, [stations], sampling_hz=sampling_hz)
    
    model = SeismicModel.ak135()
    pydsm_input = PyDSMInput.input_from_arrays(
        event, stations, model, tlen, nspc, sampling_hz)
    pydsm_output = compute(pydsm_input, mode=mode)
    pydsm_output.to_time_domain()
    dataset.data = pydsm_output.us

    return dataset
