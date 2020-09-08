from pydsm.seismicmodel import SeismicModel
from pydsm.modelparameters import ModelParameters, ParameterType
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

    def sample_one_model(self, model_id):
        value_dict = dict()
        for param_type in self.model_params._types:
            values = self.rng.uniform(
                self.range_dict[param_type][0],
                self.range_dict[param_type][1],
                self.model_params._n_nodes)
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
            windows (list(pydsm.window.Windows)): time windows. See
                pydsm.windows_from_dataset()
        Returns:
            misfit_dict (dict): values are ndarray((n_models, n_windows))
                containing misfit values (corr, variance)
        '''
        n_mod = len(models)
        n_ev = len(dataset.events)
        n_traces = dataset.nr

        corrs = np.empty((n_mod, n_traces), dtype=np.float32)
        variances = np.empty((n_mod, n_traces), dtype=np.float32)

        for imod in range(n_mod):
            win_count = 0
            for iev in range(n_ev):
                output = outputs[imod][iev]
                start, end = dataset.get_bounds_from_event_index(iev)
                data = dataset.data[:, start:end, :]
                
                output.to_time_domain()

                for i in range(end-start):
                    window = windows[win_count].to_array()
                    icomp = window.component.value
                    i_start = int(window[0] * dataset.sampling_hz)
                    i_end = int(window[1] * dataset.sampling_hz)
                    u_cut = output.us[icomp, i, i_start:i_end]
                    data_cut = data[icomp, i, i_start:i_end]

                    corr = np.corrcoef(u_cut, data_cut)[0, 1]
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
