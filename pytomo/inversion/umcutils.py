from dsmpy.seismicmodel import SeismicModel
from dsmpy.modelparameters import ModelParameters, ParameterType
import numpy as np
import matplotlib.pyplot as plt


class UniformMonteCarlo:
    """Implements the Uniform Monte Carlo method.

    Args:
        model (SeismicModel): reference seismic model.
        model_params (ModelParameters):
        range_dict (dict): entries are of type
            ParameterType:ndarray.
        mesh_type (str): 'triangle' or 'boxcar' (default is 'triangle').
        seed (int): seed for the random number generator
            (default is None).

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
        elif mesh_type == 'lininterp':
            self.model = model.lininterp_mesh(
                self.model_params, discontinuous=True)
            self.mesh = self.model.__copy__()
        else:
            raise ValueError("Expect 'triangle' or 'boxcar' or 'lininterp'")
        self.rng = np.random.default_rng(seed)

    def sample_models(self, ns):
        '''Sample ns seismic models using a uniform distribution.

        Args:
            ns (int): number of models.

        Returns:
            models (list of SeismicModel): list of sampled models.
            perturbations (dict): model perturbations. Entries are of
                type ParameterType:ndarray of shape (ns, n_grid_params).

        '''
        perturbations = []
        models = []

        free_indices = set(self.model_params.get_free_indices())
        for imod in range(ns):
            model_id = 'model_{}'.format(imod)
            value_dict = dict()

            it = 0
            for param_type in self.model_params._types:
                values = np.zeros(
                    self.model_params._n_grd_params, dtype='float')
                for igrd in range(self.model_params._n_grd_params):
                    if it not in free_indices:
                        it += 1
                        continue
                    it += 1
                    values[igrd] = self.rng.uniform(
                        self.range_dict[param_type][igrd, 0],
                        self.range_dict[param_type][igrd, 1],
                        1)

                value_dict[param_type] = values

            perturbations.append(
                np.hstack([v for v in value_dict.values()]))

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
        '''Process the output of compute_models_parallel().

        Args:
            outputs (list of list of PyDSMOutput): (n_models, n_events)
            dataset (Dataset): dataset with observed data. Same as the
                one used for input to compute_models_parallel()
            models (list of SeismicModel): seismic models
            windows (list of Window): time windows. See
                windows_from_dataset()
        Returns:
            misfit_dict (dict): values are 
                ndarray((n_models, n_windows))
                containing misfit values (corr, variance)
        '''
        n_mod = len(models)
        n_ev = len(dataset.events)
        n_window = len(windows)

        corrs = np.empty((n_mod, n_window), dtype='float')
        variances = np.empty((n_mod, n_window), dtype='float')
        rolling_variances = np.empty((n_mod, n_window), dtype='float')
        noises = np.empty((n_mod, n_window), dtype='float')
        data_norms = np.empty((n_mod, n_window), dtype='float')

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
                    jsta = np.argwhere(output.stations == station)[0][0]
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
                        noise = dataset.noise[iwin, icomp, ista]

                        if np.all(u_cut == 0):
                            print('{} {} is zero'.format(imod, window))

                        weight = 1. / np.max(np.abs(data_cut))
                        u_cut_w = u_cut * weight
                        data_cut_w = data_cut * weight
                        noise_w = noise * weight
                        corr = 0.5 * (
                                1. - np.corrcoef(u_cut_w, data_cut_w)[0, 1])
                        variance = (
                                np.dot(u_cut_w - data_cut_w,
                                       u_cut_w - data_cut_w)
                                / len(data_cut_w))
                        # / np.dot(data_cut_w, data_cut_w))
                        # rolling window variance
                        size = int(10 * dataset.sampling_hz)
                        stride = int(2 * dataset.sampling_hz)
                        rolling_variance = (UniformMonteCarlo
                            ._rolling_variance_misfit(
                            u_cut, data_cut, size, stride))
                        corrs[imod, win_count] = corr
                        variances[imod, win_count] = variance
                        rolling_variances[imod, win_count] = rolling_variance
                        noises[imod, win_count] = noise_w
                        data_norms[imod, win_count] = np.dot(
                            data_cut_w, data_cut_w)
                        win_count += 1

        misfit_dict = {
            'corr': corrs,
            'variance': variances,
            'rolling_variance': rolling_variances,
            'noise': noises,
            'data_norm': data_norms}
        return misfit_dict

    @staticmethod
    def _rolling_variance_misfit(arr1, arr2, size, stride):
        assert len(arr1) == len(arr2)
        n = int((len(arr1) - size + 1) // stride)
        indices = np.array(
            [np.arange(size) + stride * i
             for i in range(n)]
        )
        residuals = arr1[indices] - arr2[indices]
        residuals /= np.max(residuals, axis=1).reshape(-1, 1)
        return np.sum(np.sum(residuals ** 2, axis=1))


if __name__ == '__main__':
    types = [ParameterType.VSH, ParameterType.VPH]
    radii = np.array([5701., 5971.])
    model_params = ModelParameters(types, radii)
    model = SeismicModel.prem()
    range_dict = {ParameterType.VSH: [-0.15, 0.15],
                  ParameterType.VPH: [-0.1, 0.1]}

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
