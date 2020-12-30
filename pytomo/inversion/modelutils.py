from dsmpy.modelparameters import ParameterType, ModelParameters
from dsmpy.seismicmodel import SeismicModel
import numpy as np

def std_boxcar_mesh(
        n_upper_mantle=20, n_mtz=10, n_lower_mantle=12, n_dpp=6,
        types=[ParameterType.VSH], verbose=0):
    '''Boxcar mesh using ak135 as reference model for the structure of
        the upper mantle and transition zone down to 1000 km depth.
    '''
    model_ref = SeismicModel.prem()
    # model parameters
    depth_moho = 6371. - 6336.6
    depth_410 = 410.
    depth_660 = 660.
    depth_dpp = 2491.
    depth_cmb = 2891.
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
        print('dr_um={}, dr_mtz={}, dr_lm={}'.format(
            rs_upper_mantle[1] - rs_upper_mantle[0],
            rs_mtz[1] - rs_mtz[0],
            rs_lower_mantle[1] - rs_lower_mantle[0]))

    model_params = ModelParameters(types, radii, mesh_type='boxcar')
    # mesh
    # model, mesh = ak135.boxcar_mesh(model_params)
    
    return model_ref, model_params