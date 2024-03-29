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
import sys

def get_model(
        n_upper_mantle=20, n_mtz=10, n_lower_mantle=12, n_dpp=8,
        types=[ParameterType.VSH], verbose=0):
    """Boxcar mesh using ak135 as reference model for the structure of
    the upper mantle and transition zone down to 1000 km depth.
    """
    ak135 = SeismicModel.ak135()
    # model parameters
    depth_moho = 6371. - 6336.6
    depth_410 = 410.
    depth_660 = 660.
    depth_dpp = 2491.5
    depth_cmb = 2891.5
    rs_upper_mantle = np.linspace(depth_410, depth_moho, n_upper_mantle)
    rs_mtz = np.linspace(depth_660, depth_410, n_mtz,
        endpoint=(n_upper_mantle==0))
    rs_lower_mantle = np.linspace(
        depth_dpp, depth_660, n_lower_mantle, endpoint=(n_mtz==0))
    rs_dpp = np.linspace(
        depth_cmb, depth_dpp, n_dpp, endpoint=(n_lower_mantle==0))
    radii = 6371. - np.round(
        np.hstack((rs_lower_mantle, rs_mtz, rs_upper_mantle, rs_dpp)), 4)
    if verbose > 1:
        print('dr_um={}, dr_mtz={}, dr_lm={}, dr_dpp={}'.format(
            rs_upper_mantle[1] - rs_upper_mantle[0],
            rs_mtz[1] - rs_mtz[0],
            rs_lower_mantle[1] - rs_lower_mantle[0],
            rs_dpp[1] - rs_dpp[0]))

    model_params = ModelParameters(types, radii, mesh_type='boxcar')
    # mesh
    # model, mesh = ak135.boxcar_mesh(model_params)
    
    return ak135, model_params

def get_model_lininterp(
        n_upper_mantle=20, n_mtz=10, n_lower_mantle=12, n_dpp=8,
        types=[ParameterType.VSH], discontinuous=False, verbose=0):
    """Boxcar mesh using ak135 as reference model for the structure of
    the upper mantle and transition zone down to 1000 km depth.
    """
    ak135 = SeismicModel.ak135()

    # model parameters
    depth_moho = 6371. - 6336.6
    depth_410 = 410.
    depth_660 = 660.
    depth_dpp = 2691.5
    depth_cmb = 2891.5
    rs_upper_mantle = np.linspace(depth_410, depth_moho, n_upper_mantle)
    rs_mtz = np.linspace(depth_660, depth_410, n_mtz,
        endpoint=(n_upper_mantle==0))
    rs_lower_mantle = np.linspace(
        depth_dpp, depth_660, n_lower_mantle, endpoint=(n_mtz==0))
    rs_dpp = np.linspace(
        depth_cmb, depth_dpp, n_dpp, endpoint=(n_lower_mantle==0))
    radii = 6371. - np.round(
        np.hstack((rs_dpp, rs_lower_mantle, rs_mtz, rs_upper_mantle)), 4)
    if verbose > 1:
        print('dr_um={}, dr_mtz={}, dr_lm={}, dr_dpp={}'.format(
            rs_upper_mantle[1] - rs_upper_mantle[0],
            rs_mtz[1] - rs_mtz[0],
            rs_lower_mantle[1] - rs_lower_mantle[0],
            rs_dpp[1] - rs_dpp[0]))

    model_params = ModelParameters(types, radii, mesh_type='lininterp')
    # mesh
    model = ak135.lininterp_mesh(model_params, discontinuous=discontinuous)

    # set D'' layer to constant velocity and density
    idpp = model.get_zone(3479.5)
    for p_type in [
            ParameterType.VSH, ParameterType.VSV,
            ParameterType.VPH, ParameterType.VPV,
            ParameterType.RHO]:
        v_dpp = model.get_value_at(6371.-depth_dpp, p_type)
        model.set_value(
            idpp, p_type, np.array([v_dpp, 0., 0., 0.]))
    
    return model, model_params

def get_ref_event():
    catalog = read_catalog()
    event = Event.event_from_catalog(
        catalog, '200707211534A')
    event.source_time_function.half_duration = 2.
    return event

def get_ref_event2():
    catalog = read_catalog()
    event = Event.event_from_catalog(
        catalog, '200707211327A')
    event.source_time_function.half_duration = 2.
    return event

def get_dataset(
        model, tlen=1638.4, nspc=64, sampling_hz=20, mode=0,
        add_noise=False, noise_normalized_std=1., event=None):
    #TODO fix outputs.us=NaN when event.latitude==station.latitude
    if event is None:
        event = get_ref_event()
    events = [event]
    stations = [
        Station(
            '{:03d}'.format(i), 'DSM',
            event.latitude+70+0.5*i, event.longitude+0.1)
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
            dataset.data[:, :, :npts_cut]).max(axis=3, keepdims=True)
        noise_arr *= norm
        dataset.data += noise_arr

    return dataset, pydsm_output

def get_dataset_ref(tlen=1638.4, nspc=256, sampling_hz=20, mode=0):
    return get_dataset(
        SeismicModel.ak135(), tlen, nspc, sampling_hz, mode)[0]

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

def get_dataset_syntest4(
        tlen=1638.4, nspc=256, sampling_hz=20, mode=0,
        add_noise=False, noise_normalized_std=1.):
    return get_dataset(
        get_model_syntest4(), tlen, nspc, sampling_hz, mode,
        add_noise, noise_normalized_std)

def get_dataset_syntest_vshvsv_4(
        tlen=1638.4, nspc=256, sampling_hz=20, mode=0,
        add_noise=False, noise_normalized_std=1.):
    return get_dataset(
        get_model_syntest_vshvsv_4(), tlen, nspc, sampling_hz, mode,
        add_noise, noise_normalized_std)

def get_dataset_syntest_cmb_topo(
        tlen=1638.4, nspc=256, sampling_hz=20, mode=0,
        add_noise=False, noise_normalized_std=1.):
    return get_dataset(
        get_model_syntest_cmb_topo(), tlen, nspc, sampling_hz, mode,
        add_noise, noise_normalized_std)

def get_model_syntest1():
    model_ref = SeismicModel.ak135()
    types = [ParameterType.VSH]
    radii = np.array([3479.5, 3680., 3880.], dtype='float')
    model_params = ModelParameters(types, radii, mesh_type='boxcar')
    model, mesh = model_ref.boxcar_mesh(model_params)
    values = np.array(
        [.2 * (-1)**i for i in range(model_params.get_n_grd_params())])
    values_dict = {param_type: values for param_type in types}
    values_mat = model_params.get_values_matrix(values_dict)
    model_updated = mesh.multiply(values_mat)

    return model_updated

def get_model_syntest1_prem():
    types = [ParameterType.VSH]
    radii = np.array([3480. + 100 * i for i in range(5)])
    model_params = ModelParameters(types, radii, mesh_type='boxcar')
    model = SeismicModel.prem().boxcar_mesh(model_params)
    values = np.array(
        [.2 * (-1)**i if i < 2 else 0.
         for i in range(model_params.get_n_grd_params())])
    values_dict = {param_type: values for param_type in types}
    values_mat = model_params.get_values_matrix(values_dict)
    model_updated = model.multiply(values_mat)

    return model_updated

def get_model_syntest1_prem_vshvsv():
    types = [ParameterType.VSH, ParameterType.VSV]
    radii = np.array([3480. + 20 * i for i in range(21)])
    model_params = ModelParameters(types, radii, mesh_type='boxcar')
    model = SeismicModel.prem().boxcar_mesh(model_params)
    values_vsh = np.array(
        [0.2 * (-1)**(i//5) if i <= 9 else 0.
         for i in range(model_params.get_n_grd_params())])
    values_vsv = np.array(
        [-0.2 * (-1) ** (i // 5) if i <= 4 else 0.
         for i in range(model_params.get_n_grd_params())]
    )
    values_dict = {
        ParameterType.VSH: values_vsh,
        ParameterType.VSV: values_vsv
    }
    values_mat = model_params.get_values_matrix(values_dict)
    model_updated = model.multiply(values_mat)

    return model_updated


def get_model_syntest2_prem_vshvsv():
    types = [ParameterType.VSH, ParameterType.VSV]
    radii = np.array([3480. + 20 * i for i in range(21)])
    model_params = ModelParameters(types, radii, mesh_type='boxcar')
    model = SeismicModel.prem().boxcar_mesh(model_params)
    values_vsh = np.array(
        [0.15 if i <= 9 else 0.
         for i in range(model_params.get_n_grd_params())])
    values_vsv = np.array(
        [0. if i <= 4 else 0.
         for i in range(model_params.get_n_grd_params())]
    )
    values_dict = {
        ParameterType.VSH: values_vsh,
        ParameterType.VSV: values_vsv
    }
    values_mat = model_params.get_values_matrix(values_dict)
    model_updated = model.multiply(values_mat)

    return model_updated


def get_model_syntest2_prem_vshqmu():
    types = [ParameterType.VSH, ParameterType.QMU]
    radii = np.array([3480. + 20 * i for i in range(21)])
    model_params = ModelParameters(types, radii, mesh_type='boxcar')
    model = SeismicModel.prem().boxcar_mesh(model_params)
    values_vsh = np.array(
        [0.15 if i <= 9 else 0.
         for i in range(model_params.get_n_grd_params())])
    values_qmu = np.array(
        [150 if i <= 9 else 0.
         for i in range(model_params.get_n_grd_params())]
    )
    values_dict = {
        ParameterType.VSH: values_vsh,
        ParameterType.QMU: values_qmu
    }
    values_mat = model_params.get_values_matrix(values_dict)
    model_updated = model.multiply(values_mat)

    return model_updated


def get_model_syntest3_prem_vshqmu():
    types = [ParameterType.VSH, ParameterType.QMU]
    radii = np.array([3480. + 20 * i for i in range(21)])
    model_params = ModelParameters(types, radii, mesh_type='boxcar')
    model = SeismicModel.prem().boxcar_mesh(model_params)
    values_vsh = np.array(
        [0. if i <= 9 else 0.
         for i in range(model_params.get_n_grd_params())])
    values_qmu = np.array(
        [150 for i in range(model_params.get_n_grd_params())]
    )
    values_dict = {
        ParameterType.VSH: values_vsh,
        ParameterType.QMU: values_qmu
    }
    values_mat = model_params.get_values_matrix(values_dict)
    model_updated = model.multiply(values_mat)

    return model_updated


def get_model_syntest2():
    model_ref = SeismicModel.ak135()
    types = [ParameterType.VSH]
    radii = np.array([3479.5, 3679.5], dtype='float')
    model_params = ModelParameters(
        types, radii, mesh_type='lininterp')
    model = model_ref.lininterp_mesh(
        model_params, discontinuous=True)
    
    # set D'' layer to constant velocity and density
    depth_dpp = 2691.5
    idpp = model.get_zone(3479.5)
    for p_type in [
            ParameterType.VSH, ParameterType.VSV,
            ParameterType.VPH, ParameterType.VPV,
            ParameterType.RHO]:
        v_dpp = model.get_value_at(6371.-depth_dpp, p_type)
        model.set_value(
            idpp, p_type, np.array([v_dpp, 0., 0., 0.]))

    values = np.array([0.2, 0.2, 0.2, 0.2])
    values_dict = {param_type: values for param_type in types}
    model_mul = model.build_model(model, model_params, values_dict)

    return model_mul

def get_model_syntest3():
    model_ref = SeismicModel.ak135()
    types = [ParameterType.VSH]
    radii = np.array([3479.5, 3529.5, 3679.5], dtype='float')
    model_params = ModelParameters(
        types, radii, mesh_type='lininterp')
    model = model_ref.lininterp_mesh(
        model_params, discontinuous=True)

    izone = model.get_zone(3479.5)
    vsh_dpp = model.get_value_at(radii[-1], ParameterType.VSH)
    model._vsh[:, izone] = np.array([vsh_dpp, 0., 0., 0.])
    model._vsh[:, izone+1] = np.array([vsh_dpp, 0., 0., 0.])

    values_p = np.array([0., 0.2, 0.2])
    values_dict_p = {param_type: values_p for param_type in types}
    values_m = np.array([0., 0., 0.2])
    values_dict_m = {param_type: values_m for param_type in types}
    model_mul = model.build_model(
        model, model_params, values_dict_p, values_dict_m)

    return model_mul

def get_model_syntest4():
    model_ref = SeismicModel.ak135()
    types = [ParameterType.VSH]
    radii = np.array([3479.5, 3489.5], dtype='float')
    model_params = ModelParameters(
        types, radii, mesh_type='lininterp')
    model = model_ref.lininterp_mesh(
        model_params, discontinuous=True)

    izone = model.get_zone(3479.5)
    vsh_dpp = model.get_value_at(radii[-1], ParameterType.VSH)
    model._vsh[:, izone] = np.array([vsh_dpp, 0., 0., 0.])
    # model._vsh[:, izone+1] = np.array([vsh_dpp, 0., 0., 0.])

    values_p = np.array([0.2, 0.2])
    values_dict_p = {param_type: values_p for param_type in types}
    values_m = np.array([0., 0.])
    values_dict_m = {param_type: values_m for param_type in types}
    model_mul = model.build_model(
        model, model_params, values_dict_p)

    return model_mul

def get_model_syntest_vshvsv_4():
    model_ref = SeismicModel.prem()
    types = [ParameterType.VSH, ParameterType.VSV]
    radii = np.linspace(3480., 3980., 3, endpoint=True)
    model_params = ModelParameters(
        types, radii, mesh_type='lininterp')
    model = model_ref.lininterp_mesh(
        model_params)

    values = np.zeros(model_params.get_n_grd_params())
    values[2] = model_ref.get_value_at(radii[1], ParameterType.VSH) * 0.03
    value_dict = {
        ParameterType.VSH: values,
        ParameterType.VSV: -values
    }
    model_updated = model.build_model(
        model, model_params, value_dict)

    return model_updated

def get_model_syntest5():
    model_ref = SeismicModel.ak135()
    types = [ParameterType.VSH]
    radii = np.array(
        [3479.5+i*50 for i in range(9)], dtype='float')
    model_params = ModelParameters(types, radii, mesh_type='boxcar')
    model, mesh = model_ref.boxcar_mesh(model_params)
    values = np.array(
        [.2 * (-1)**i for i in range(model_params._n_grd_params)])
    values_dict = {param_type: values for param_type in types}
    values_mat = model_params.get_values_matrix(values_dict)
    mesh_mul = mesh.multiply(model_params.get_nodes(), values_mat)
    model_mul = model + mesh_mul

    return model_mul

def get_model_syntest_cmb_topo():
    model_ref = SeismicModel.ak135()
    types = [ParameterType.VSV, ParameterType.RADIUS]
    radii = np.array([3479.5, 3679.5], dtype='float')
    model_params = ModelParameters(
        types, radii, mesh_type='lininterp')
    model = model_ref.lininterp_mesh(
        model_params, discontinuous=True)
    
    # set D'' layer to constant velocity and density
    depth_dpp = 2691.5
    idpp = model.get_zone(3479.5)
    for p_type in [
            ParameterType.VSH, ParameterType.VSV,
            ParameterType.VPH, ParameterType.VPV,
            ParameterType.RHO]:
        v_dpp = model.get_value_at(6371.-depth_dpp, p_type)
        model.set_value(
            idpp, p_type, np.array([v_dpp, 0., 0., 0.]))

    values = np.array([0.2, 0.2])
    values_dict = {
        ParameterType.VSV: values,
        ParameterType.RADIUS: np.array([-8., 0.])}
    model_mul = model.build_model(model, model_params, values_dict)

    return model_mul


if __name__ == '__main__':
    model_syntest2 = get_model_syntest2()
    fig, ax = model_syntest2.plot()
    SeismicModel.ak135().plot(ax=ax, label='ak135')
    plt.show()
    # plt.savefig('model_syntest2.pdf')
    # plt.show()
    # plt.close(fig)

    # fig, ax = get_model_syntest4().plot(types=[ParameterType.VSH])
    # SeismicModel.ak135().plot(types=[ParameterType.VSH], ax=ax, label='ak135')
    # plt.show()

    model_, model_params = get_model_lininterp(0, 0, 0, 2, discontinuous=True)

    value_dict = {
        ParameterType.VSH: np.array([0.2, 0.2]),
        ParameterType.RADIUS: np.array([0., -8.])}
    value_dict_m = {
        ParameterType.VSH: np.array([0.2, 0.2]),
        ParameterType.RADIUS: np.array([0., 0])}
    model = model_.build_model(model_, model_params, value_dict, value_dict_m)
    fig, ax = model.plot(types=model_params._types, label='1')
    print(model._vrmin)
    print(model._vrmax)
    plt.show()

    value_dict = {
        ParameterType.VSH: np.array([0., 0.2, 0.2]),
        ParameterType.RADIUS: np.array([0., -100, 0.])}
    value_dict_m = {
        ParameterType.VSH: np.array([0., 0., 0.2]),
        ParameterType.RADIUS: np.array([0., -100, 0.])}
    model = model_.build_model(model_, model_params, value_dict, value_dict_m)
    print(model._vrmin)
    print(model._vrmax)
    model.plot(types=model_params._types, label='2', ax=ax)

    value_dict = {
        ParameterType.VSH: np.array([0., 0.2, 0.2]),
        ParameterType.RADIUS: np.array([0., -60., 0.])}
    value_dict_m = {
        ParameterType.VSH: np.array([0., 0., 0.2]),
        ParameterType.RADIUS: np.array([0., -60., 0.])}
    model = model_.build_model(model_, model_params, value_dict, value_dict_m)
    model.plot(types=model_params._types, label='3', ax=ax)

    value_dict = {
        ParameterType.VSH: np.array([0., 0., 0.]),
        ParameterType.RADIUS: np.array([0., 0., 0.])}
    value_dict_m = {
        ParameterType.VSH: np.array([0., 0., 0.]),
        ParameterType.RADIUS: np.array([0., 0., 0.])}
    model = model_.build_model(model_, model_params, value_dict, value_dict_m)
    model.plot(types=model_params._types, label='ref', ax=ax)
    
    # SeismicModel.ak135().plot(types=[ParameterType.VSH], ax=ax, label='ak135')
    get_model_syntest3().plot(types=[ParameterType.VSH], ax=ax, label='target')
    plt.show()
    plt.close(fig)

    # dataset, output = get_dataset_syntest1(
    #     mode=2, add_noise=True, noise_normalized_std=1.)
    # dataset.filter(0.005, 0.1, type='bandpass')
    # dataset.plot_event(0, color='black')
    # plt.show()
