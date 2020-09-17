from pydsm.seismicmodel import SeismicModel
from pydsm.modelparameters import ModelParameters, ParameterType
from pytomo.work.ca import params as work_params
import numpy as np
import matplotlib.pyplot as plt
import time

class UniformMonteCarlo:
    """Implements the uniform monte carlo method.

    Args:
        model (SeismicModel):
        model_params (ModelParameters):
        range_dict (dict): ParameterType:ndarray((2,))
        mesh_type (str): 'triangle' or 'boxcar' (default: 'triangle')
        seed (int): seed for the random number generator (default: None)
    """

    def __init__(
            self, model, model_params, range_dict,
            mesh_type='boxcar', seed=None):
        self.model_params = model_params
        self.range_dict = range_dict
        if mesh_type == 'triangle':
            self.model, self.mesh = model.triangle_mesh(self.model_params)
        elif mesh_type == 'boxcar':
            self.model, self.mesh = model.boxcar_mesh(self.model_params)
        else:
            raise ValueError("Expect 'triangle' or 'boxcar'")
        self.rng = np.random.default_rng(seed)

    def sample_models(self, ns):
        '''
        Returns:
            models (list): list of ns sampled models
            perturbations (dict): ndarray(ns, n_grid_params)
        '''

        perturbations = dict()
        models = []

        for param_type in self.model_params._types:
            perturbations[param_type] = np.zeros(
                (ns, self.model_params._n_grd_params), dtype='float')

        for imod in range(ns):
            model_id = 'model_{}'.format(imod)
            value_dict = dict()
            for param_type in self.model_params._types:
                values = np.zeros(
                    self.model_params._n_grd_params, dtype='float')
                for igrd in range(self.model_params._n_grd_params):
                    values[igrd] = self.rng.uniform(
                        self.range_dict[param_type][igrd, 0],
                        self.range_dict[param_type][igrd, 1],
                        1)
                value_dict[param_type] = values
                perturbations[param_type][imod] = values
            model_sample = self.model.build_model(
                self.mesh, self.model_params, value_dict)
            model_sample._model_id = model_id
            models.append(model_sample)

        return models, perturbations

    def get_best_models(self, misfit_dict, n_best):
        '''Get the n_best best models'''
        avg_var = misfit_dict['variance'].mean(axis=1)
        indices_best = np.argsort(avg_var)[:n_best]
        return indices_best

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
                
                # TODO using to_time_domain() erase the effect of filtering
                # output.to_time_domain()

                for ista in range(start, end):
                    station = dataset.stations[ista]
                    jsta = np.argwhere(output.stations==station)[0][0]
                    windows_filt = [
                        window for window in windows
                        if (window.station == station
                            and window.event == event)]
                    for iwin, window in enumerate(windows_filt):
                        window_arr = window.to_array()
                        icomp = window.component.value
                        i_start = int(window_arr[0] * dataset.sampling_hz)
                        i_end = int(window_arr[1] * dataset.sampling_hz)
                        u_cut = output.us[icomp, jsta, i_start:i_end]
                        data_cut = dataset.data[
                            iwin, icomp, ista, :]

                        if np.all(u_cut==0):
                            print('{} {} is zero'.format(imod, window))

                        corr = 0.5 * (1. - np.corrcoef(u_cut, data_cut)[0, 1])
                        variance = (np.dot(u_cut-data_cut, u_cut-data_cut)
                            / np.dot(data_cut, data_cut))
                        corrs[imod, win_count] = corr
                        variances[imod, win_count] = variance

                        win_count += 1
                    
        misfit_dict = {'corr': corrs, 'variance': variances}
        return misfit_dict


if __name__ == '__main__':
    types = [ParameterType.VSH, ParameterType.VPH]
    radii = np.array([5701., 5971.])
    model_params = ModelParameters(types, radii)
    model = SeismicModel.prem()
    range_dict = {ParameterType.VSH:[-0.15,0.15],
                 ParameterType.VPH:[-0.1,0.1]}

    umc = UniformMonteCarlo(
        model, model_params, range_dict,
        mesh_type='triangle', seed=0)
    sample_models = umc.sample_models(50)

    fig, ax = sample_models[0].plot(
        types=[ParameterType.VSH, ParameterType.VPH])
    for sample in sample_models[1:]:
        sample.plot(
            ax=ax, types=[ParameterType.VSH, ParameterType.VPH])
    ax.get_legend().remove()
    plt.show()
