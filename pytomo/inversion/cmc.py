from pydsm.seismicmodel import SeismicModel
from pydsm.modelparameters import ModelParameters, ParameterType
import numpy as np
import matplotlib.pyplot as plt
import time
from pytomo.work.jp.params import get_model
import os
import glob

class InputFile:
    """Input file for ConstrainedMonteCarlo (cmc).

    Args:
        input_file (str): path of cmc input file
    """
    def __init__(self, input_file):
        self.input_file = input_file
    
    def read(self):
        params = dict()
        params['verbose'] = 0
        params['filter_type'] = None
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
            value_parsed = list(glob.iglob(full_path))
        elif key == 'tlen':
            value_parsed = float(value)
        elif key == 'nspc':
            value_parsed = int(value)
        elif key == 'sampling_hz':
            value_parsed = int(value)
        elif key == 'n_mod':
            value_parsed = int(value)
        elif key == 'n_block':
            value_parsed = int(value)
        elif key == 'mode':
            value_parsed = int(value)
        elif key == 'result_path':
            full_path = os.path.expanduser(value.strip())
            value_parsed = full_path
        elif key == 'verbose':
            value_parsed = int(value)
        elif key == 'freq':
            value_parsed = float(value)
        elif key == 'freq2':
            value_parsed = float(value)
        elif key == 'filter_type':
            value_parsed = value.strip().lower()
        else:
            print('Warning: key {} undefined. Ignoring.'.format(key))
            return None, None
        return key, value_parsed

class ConstrainedMonteCarlo:
    """Implements a constrained monte carlo method.

    Args:
        model (SeismicModel):
        model_params (ModelParameters):
        cov (2-D ndarray): covariance matrix
        mesh_type (str): 'triangle' or 'boxcar' (default: 'triangle')
        seed (int): seed for the random number generator (default: None)
    """

    def __init__(
            self, model, model_params, cov,
            mesh_type='boxcar', seed=None):
        self.model_params = model_params
        self.cov = cov
        if mesh_type == 'triangle':
            self.model, self.mesh = model.triangle_mesh(self.model_params)
        elif mesh_type == 'boxcar':
            self.model, self.mesh = model.boxcar_mesh(self.model_params)
        else:
            raise ValueError("Expect 'triangle' or 'boxcar'")
        self.rng = np.random.default_rng(seed)

    def sample_one_model(self, model_id):
        value_dict = dict()
        mean = np.zeros(self.model_params._n_nodes)
        for param_type in self.model_params._types:
            values = self.rng.multivariate_normal(
                mean, self.cov)
            value_dict[param_type] = values
        model_sample = self.model.build_model(
            self.mesh, self.model_params, value_dict)
        model_sample._model_id = model_id
        return model_sample

    def sample_models(self, ns):
        models = [self.sample_one_model('model_{}'.format(i))
                  for i in range(ns)]
        return models

    def process_outputs(self, outputs, dataset, models, windows):
        '''Process the output of pydsm.dsm.compute_models_parallel.
        Args:
            outputs (list(list(PyDSMOutput))): "shape" = (n_models, n_events)
            dataset (pydsm.Dataset): dataset with observed data. Same as the
                one used for input to compute_models_parallel()
            models (list(pydsm.SeismicModel)): seismic models
            windows (list(pydsm.window.Window)): time windows. See
                pydsm.windows_from_dataset()
        Returns:
            misfit_dict (dict): values are ndarray((n_models, n_windows))
                containing misfit values (corr, variance)
        '''
        n_mod = len(models)
        n_ev = len(dataset.events)
        n_window = len(windows)

        corrs = np.empty((n_mod, n_window), dtype=np.float32)
        variances = np.empty((n_mod, n_window), dtype=np.float32)

        for imod in range(n_mod):
            win_count = 0
            for iev in range(n_ev):
                event = dataset.events[iev]
                output = outputs[imod][iev]
                start, end = dataset.get_bounds_from_event_index(iev)
                data = dataset.data
                
                output.to_time_domain()

                for ista in range(start, end):
                    station = dataset.stations[ista]
                    jsta = np.argwhere(output.stations==station)[0][0]
                    windows_filt = [
                        window for window in windows
                        if (window.station == station
                            and window.event == event)]
                    for window in windows_filt:
                        window_arr = window.to_array()
                        icomp = window.component.value
                        i_start = int(window_arr[0] * dataset.sampling_hz)
                        i_end = int(window_arr[1] * dataset.sampling_hz)
                        u_cut = output.us[icomp, jsta, i_start:i_end]
                        data_cut = dataset.data[icomp, ista, i_start:i_end]

                        corr = np.corrcoef(u_cut, data_cut)[0, 1]
                        variance = (np.dot(u_cut-data_cut, u_cut-data_cut)
                            / np.dot(data_cut, data_cut))
                        corrs[imod, win_count] = corr
                        variances[imod, win_count] = variance

                        win_count += 1

        misfit_dict = {'corr': corrs, 'variance': variances}
        return misfit_dict

    @staticmethod
    def smooth_damp_cov(n, g, l):
        '''Compute the precision matrix (inverse of covariance)
        to impose smoothness and damping on normally distributed models.
        Args:
            n (int): number of model parameters
            g (float): coefficient for smoothing (larger is smoother)
            l (float): coefficient for damping (larger is more damped)
        Return:
            prec (ndarray((n,n))): precision matrix
        '''
        G = np.zeros((n, n), np.float32)
        L = np.zeros((n, n), np.float32)

        for i in range(n):
            L[i, i] = l
            G[i, i] = -2 * g
        for i in range(n-1):
            G[i+1, i] = g
            G[i, i+1] = g

        # empirical scaling factors
        L *= 3.
        G *= 4.

        cov = L.T@L + G.T@G
        # precision matrix is the inverse of the covariance matrix
        prec = np.linalg.inv(cov)
        
        return prec


if __name__ == '__main__':
    # types = [ParameterType.VSH, ParameterType.VPH]
    # radii = np.array([5701., 5971.])
    # model_params = ModelParameters(types, radii)
    # model = SeismicModel.prem()
    
    types = [ParameterType.VSH]
    model, model_params = get_model(types=types)

    n = model_params._n_nodes
    l = 1.
    g = 1.
    cov = ConstrainedMonteCarlo.smooth_damp_cov(n, g, l)

    seed = time.time_ns()

    cmc = ConstrainedMonteCarlo(
        model, model_params, cov,
        mesh_type='boxcar', seed=seed)
    sample_models = cmc.sample_models(2)

    fig, ax = sample_models[0].plot(types=[ParameterType.VSH])
    for sample in sample_models[1:]:
        sample.plot(ax=ax, types=[ParameterType.VSH])
    ax.get_legend().remove()
    ax.set_ylim(model_params._radii[0] - 100, 6371.)
    plt.show()
