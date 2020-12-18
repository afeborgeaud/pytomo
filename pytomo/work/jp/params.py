from dsmpy.modelparameters import ModelParameters, ParameterType
from dsmpy.seismicmodel import SeismicModel
from dsmpy.station import Station
from dsmpy.event import Event
from dsmpy.utils.cmtcatalog import read_catalog
from dsmpy.dataset import Dataset
from dsmpy.dsm import PyDSMInput, compute
from pytomo.utilities import white_noise
import numpy as np
import matplotlib.pyplot as plt

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

def get_model_lininterp(
        n_mtz=10, n_lower_mantle=12,
        types=[ParameterType.VSH], discontinuous=True, verbose=0):
    """Boxcar mesh using ak135 as reference model for the structure of
        the upper mantle and transition zone down to 1000 km depth.
    """
    ak135 = SeismicModel.ak135()
    radii = np.array(
        [5371, 5611, 5711, 5836, 5961, 6161, 6251, 6336.6])

    model_params = ModelParameters(types, radii, mesh_type='lininterp')
    # mesh
    model = ak135.lininterp_mesh(model_params, discontinuous=discontinuous)
    
    return model, model_params

def get_ref_event():
    catalog = read_catalog()
    event = Event.event_from_catalog(
        catalog, '200707211534A')
    event.source_time_function.half_duration = 2.
    return event

def get_dataset(
        model, tlen=1638.4, nspc=64, sampling_hz=20, mode=0,
        add_noise=False, noise_normalized_std=1.):
    #TODO fix outputs.us=NaN when event.latitude==station.latitude
    event = get_ref_event()
    events = [event]
    stations = [
        Station(
            '{:03d}'.format(i), 'DSM',
            event.latitude+5+0.5*i, event.longitude+0.1)
        for i in range(61)]
    dataset = Dataset.dataset_from_arrays(
        events, [stations], sampling_hz=sampling_hz)
    
    pydsm_input = PyDSMInput.input_from_arrays(
        event, stations, model, tlen, nspc, sampling_hz)
    pydsm_output = compute(pydsm_input, mode=mode)
    pydsm_output.to_time_domain()
    dataset.data = np.zeros((1,)+pydsm_output.us.shape, dtype=np.float64)
    dataset.data[0] = pydsm_output.us

    if add_noise:
        noise_arr = white_noise(
            noise_normalized_std, dataset.data.shape)
        npts_cut = int(dataset.data.shape[3]*0.9)
        norm = np.abs(
            dataset.data[:,:,:npts_cut]).max(axis=3, keepdims=True)
        noise_arr *= norm
        dataset.data += noise_arr

    return dataset, pydsm_output

def get_model_syntest1():
    model_ref = SeismicModel.ak135()
    types = [ParameterType.VSH]
    radii = 6371. - np.array([493.33, 410.])
    model_params = ModelParameters(types, radii, mesh_type='boxcar')
    model, mesh = model_ref.boxcar_mesh(model_params)
    values = np.array(
        [.2 * (-1)**i for i in range(model_params._n_grd_params)])
    values_dict = {param_type: values for param_type in types}
    values_mat = model_params.get_values_matrix(values_dict)
    mesh_mul = mesh.multiply(model_params.get_nodes(), values_mat)
    model_mul = model + mesh_mul

    return model_mul

def get_model_syntest2():
    types = [ParameterType.VSH]
    model_ref, model_params = get_model(
        20, 10, 12, types=types)
    model, mesh = model_ref.boxcar_mesh(model_params)
    values = np.array(
        [.2 * (-1)**i for i in range(model_params._n_grd_params)])
    values_dict = {param_type: values for param_type in types}
    values_mat = model_params.get_values_matrix(values_dict)
    mesh_mul = mesh.multiply(model_params.get_nodes(), values_mat)
    model_mul = model + mesh_mul

    return model_mul

def get_model_syntest3():
    types = [ParameterType.VSH, ParameterType.RADIUS]
    model, model_params = get_model_lininterp(
        types=types, verbose=0, discontinuous=True)

    values_vsh = np.zeros(model_params._n_grd_params)
    values_r =  np.array(
        [0, 0, 0, 0, 30., 30., 0, 0, -30, -30, 0, 0, 0, 0, 0, 0])

    values_dict = {
        ParameterType.VSH: values_vsh,
        ParameterType.RADIUS: values_r}
    model_mul = model.build_model(
        model, model_params, values_dict)

    return model_mul

def get_dataset_syntest1(
        tlen=1638.4, nspc=256, sampling_hz=20, mode=0,
        add_noise=False, noise_normalized_std=1.):
    return get_dataset(
        get_model_syntest1(), tlen, nspc, sampling_hz, mode,
        add_noise, noise_normalized_std)

def get_dataset_syntest2(
        tlen=1638.4, nspc=256, sampling_hz=20, mode=0,
        add_noise=False, noise_normalized_std=1.):
    return get_dataset(
        get_model_syntest2(), tlen, nspc, sampling_hz, mode,
        add_noise, noise_normalized_std)

def get_dataset_syntest3(
        tlen=1638.4, nspc=256, sampling_hz=20, mode=0,
        add_noise=False, noise_normalized_std=1.):
    return get_dataset(
        get_model_syntest3(), tlen, nspc, sampling_hz, mode,
        add_noise, noise_normalized_std)

if __name__ == '__main__':
    model, model_params = get_model_lininterp(
        types=[ParameterType.VSH, ParameterType.RADIUS])

    fig, ax = get_model_syntest3().plot()
    SeismicModel.ak135().plot(ax=ax)
    plt.show()

    dataset, _ = get_dataset_syntest3(nspc=64, mode=2)
    dataset.plot_event(0)
    plt.show()
    
    fig, ax = SeismicModel.ak135().plot(
        types=[ParameterType.VSH], dr=.5, label='')

    mask_dict = dict()
    mask_dict[ParameterType.VSH] = np.ones(
        model_params._n_grd_params//2, dtype='bool')
    mask_dict[ParameterType.RADIUS] = np.zeros(
        model_params._n_grd_params//2, dtype='bool')
    mask_dict[ParameterType.VSH][0] = False
    mask_dict[ParameterType.VSH][-1] = False
    for i in range(2,5):
        mask_dict[ParameterType.RADIUS][i] = True

    discon_arr = np.zeros(
        model_params._n_grd_params//2, dtype='bool')
    discon_arr[2] = True
    discon_arr[4] = True

    model_params.set_constraints(
        mask_dict=mask_dict,
        discon_arr=discon_arr)

    values_vsh = np.array(
        [0.2*(-1)**(i//2) for i in range(model_params._n_grd_params)])
    values_r =  np.array(
        [0. for i in range(model_params._n_grd_params)])

    values_dict = {
        ParameterType.VSH: values_vsh,
        ParameterType.RADIUS: values_r}
    model_mul = model.build_model(
        model, model_params, values_dict)

    print(model_params.get_n_params())
    print(model_params.get_free_indices())

    model_mul.plot(
        types=[ParameterType.VSH], dr=1., label='model', ax=ax
    )

    plt.show()
    plt.close(fig)
