from dsmpy.dsm import compute_models_parallel
from dsmpy.component import Component
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import pickle


class InversionResult:
    """Pack the inversion results, models and dataset needed to
    reproduce the results.

    Args:
        dataset (Dataset): dataset
        windows (list of Window): time windows. The order
            is the same as for axis 1 of misfit_dict.
        meta (dict): dict with meta parameters

    """

    def __init__(self, dataset, windows, meta=None):
        self.dataset = dataset
        self.windows = windows
        self.meta = meta
        self.misfit_dict = dict()
        self.models = []
        self.perturbations = []

    def add_result(self, models, misfit_dict, perturbations):
        """Add misfit_dict for new models to current inversion result.
        The dataset and windows must be the same.

        Args:
            models (list of SeismicModel): seismic models. The order
            is the same as for axis 0 of misfit_dict.
            misfit_dict (dict): values are
                ndarray((n_models, n_windows)) containing misfit
                values (corr, variance)
            perturbations (ndarray): perturbations

        """
        for key in misfit_dict.keys():
            assert len(models) == misfit_dict[key].shape[0]

        self.models += models
        self.perturbations += perturbations

        for misfit_name in misfit_dict.keys():
            if misfit_name not in self.misfit_dict:
                self.misfit_dict[misfit_name] = misfit_dict[misfit_name]
            else:
                self.misfit_dict[misfit_name] = np.vstack(
                    (self.misfit_dict[misfit_name],
                     misfit_dict[misfit_name]))

    def get_model_perturbations(
            self, smooth=True, n_s=None,
            in_percent=False):
        """Get perturbations from model_ref.

        Args:
            smooth (bool): smooth over n_s (default is True).
            n_s (int): number of models computed at each iteration
                (default is None).
            in_percent (bool): returns perturbations as percent
                (default is False).

        Returns:
            ndarray: array of perturbations.
                If smooth, shape is (n_iteration,),
                else shape is (n_models,)

        """
        perturbations = np.array(self.perturbations)

        if smooth:
            n_it = int(len(self.models) // n_s)
            if len(self.models) % n_s != 0:
                n_it += 1
            perturbations_ = np.zeros((n_it, perturbations.shape[1]))
            for i in range(n_it):
                s = i * n_s
                e = s + n_s
                perturbations_[i] = perturbations[s:e].mean(axis=0)
            perturbations = perturbations_

        return perturbations

    def get_model_perturbations_diff(
            self, n_r, scale=None, smooth=True, n_s=None):
        perturbations = np.array(self.perturbations)
        if scale is not None:
            perturbations = np.true_divide(
                perturbations, scale,
                out=np.zeros_like(perturbations),
                where=(scale != 0))
        perturbations_diff = np.abs(np.diff(perturbations, axis=0))
        mask = [(i + 1) % n_r == 0 for i in range(perturbations_diff.shape[0])]
        perturbations_diff[mask, :] = 0.

        if smooth:
            n_it = int(len(self.models) // n_s)
            if len(self.models) % n_s != 0:
                n_it += 1
            perturbations_diff_smooth = np.zeros(
                (n_it, perturbations.shape[1]))
            for i in range(n_it):
                s = i * n_s
                e = s + n_s
                perturbations_diff_smooth[i] = (
                    perturbations_diff[s:e].mean(axis=0))
            perturbations_diff = perturbations_diff_smooth

        return perturbations_diff

    def get_variances(self, key='variance', smooth=True, n_s=None):
        """Get misfit array.

        Args:
            key (object): misfit type
                ('variance', 'corr', 'rolling_variance')
                (default is 'variance').
            smooth (bool): smooth over n_s (default is True).
            n_s (int): number of models computed at each iteration.

        Returns:
            ndarray: misfit values

        """
        variances = self.misfit_dict[key].mean(axis=1)
        if smooth:
            if len(variances) % n_s != 0:
                n = int(len(variances) // n_s * n_s + n_s)
                perturbations = np.pad(
                    self.perturbations, (0, n), 'constant',
                    constant_values=(0, 0))
            variances = variances.reshape((-1, n_s)).mean(axis=1)
        return variances

    def save(self, path):
        """Save self using pickle.dump().

        Args:
            path (str): name of the output file.

        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        """Read file into self using pickle.load().

        Args:
            path (str): name of the file that contains self.

        """
        with open(path, 'rb') as f:
            output = pickle.load(f)
        return output

    def plot_models(
            self, types, n_best=-1, n_mod=-1, ax=None,
            key='variance', **kwargs):
        """Plot models colored by misfit value.

        Args:
            types (list of ParameterType):
                parameter types (e.g., ParameterType.RHO).
            n_best (int): number of best models.
            n_mod (int): number of models.
            ax (Axes): matplotlib Axes object.
            key (str): 'variance', 'corr', or 'rolling_variance'
                (default is 'variance').

        Returns:
            figures: matplotlib figure object.
            Axes: matplotlib Axes object.

        """
        assert key in {'variance', 'misfit', 'rolling_variance'}
        avg_misfit = self.misfit_dict[key].mean(axis=1)
        indices_best = np.arange(len(avg_misfit), dtype=int)
        if type(n_best) == int and n_best > 0:
            indices_best = self.get_indices_best_models(n_best, n_mod, key)
            print(avg_misfit[indices_best])
            print(avg_misfit.max())
            print(indices_best)
        elif type(n_best) == float:
            if n_best <= 1:
                # TODO make it a percentage of the number of models
                indices_best = (np.where(avg_misfit
                                         <= (1 + n_best) * avg_misfit.min()))

        cm = plt.get_cmap('Greys_r')
        c_norm = colors.Normalize(vmin=avg_misfit.min(),
                                  vmax=avg_misfit.max() * 1.1)
        scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=cm)

        if 'color' not in kwargs:
            color = scalar_map.to_rgba(avg_misfit[indices_best[0]])
        else:
            color = kwargs['color']
            kwargs.pop('color', None)

        if ax is None:
            fig, ax = self.models[indices_best[0]].plot(
                types=types, color=color, **kwargs)
        else:
            self.models[indices_best[0]].plot(
                types=types, ax=ax, color=color, **kwargs)
            fig = None

        for i in indices_best[1:]:
            color = scalar_map.to_rgba(avg_misfit[i])
            self.models[i].plot(ax=ax, types=types, color=color, **kwargs)

        # model_ref.plot(ax=ax, types=[ParameterType.VPV], color='red')
        ax.get_legend().remove()

        return fig, ax

    def get_indices_best_models(self, n_best=-1, n_mod=-1, key='variance'):
        assert key in {'corr', 'variance', 'rolling_variance'}
        avg_misfit = self.misfit_dict[key].mean(axis=1)
        n_best = min(n_best, len(avg_misfit))
        n_mod = len(avg_misfit) if n_mod == -1 else n_mod
        i_bests = avg_misfit[:n_mod].argsort()[:n_best]
        return i_bests

    def compute_models(self, models, comm):
        outputs = compute_models_parallel(
            self.dataset, models, self.dataset.tlen,
            self.dataset.nspc, self.dataset.sampling_hz,
            comm, mode=self.meta['mode'], verbose=self.meta['verbose'])
        filter_type = self.meta['filter_type']
        freq = self.meta['freq']
        freq2 = self.meta['freq2']
        if comm.Get_rank() == 0:
            if filter_type is not None:
                for imod in range(len(outputs)):
                    for iev in range(len(outputs[0])):
                        outputs[imod][iev].filter(
                            freq, freq2, filter_type)
        return outputs

    def plot_event(
            self, outputs, iev, ax, component=Component.T,
            color='cyan', **kwargs):
        outputs[iev].plot_component(
            component, self.windows, ax=ax,
            align_zero=True, color='black')
        if ('distance_min' in self.meta.keys()
                and 'distance_max' in self.meta.keys()):
            self.dataset.plot_event(
                iev, self.windows, align_zero=True,
                component=component, ax=ax,
                dist_min=self.meta['distance_min'],
                dist_max=self.meta['distance_max'],
                color=color)
        else:
            self.dataset.plot_event(
                iev, self.windows, align_zero=True,
                component=component, ax=ax, color=color)

    def save_convergence_curve(
            self, path, scale_arr, free_indices, smooth=True):
        """Save the convergence curve to path.

        Args:
             path (str): path of the output figure.
             scale_arr (ndarray):
             free_indices (list of int):
             smooth (bool): average the variance points over each
                iteration of the NA algorithm (n_s points)
                (default is True).

        """
        perturbations_diff = self.get_model_perturbations_diff(
            self.meta['n_r'], scale=scale_arr,
            smooth=True, n_s=self.meta['n_s'])
        perturbations_diff_free = perturbations_diff[
                                  :, free_indices].max(axis=1)

        x = np.arange(1, perturbations_diff_free.shape[0] + 1)

        fig, ax = plt.subplots(1)
        ax.plot(x, perturbations_diff_free, '-xk')
        if smooth:
            ax.set(
                xlabel='Iteration #',
                ylabel='Mean model perturbation (km/s)')
        else:
            ax.set(
                xlabel='Model #',
                ylabel='Model perturbation (km/s)')
        plt.savefig(path, bbox_inches='tight')
        plt.close(fig)

    def save_variance_curve(self, path, smooth=True):
        """Save the variance curve to path.

        Args:
            path (str): path to the output figure.
            smooth (bool): average the variance points over each
                iteration of the NA algorithm (n_s points)
                (default is True).

        """
        variances = self.get_variances(
            smooth=smooth, n_s=self.meta['n_s'],
            key=self.meta['misfit_type'])
        x = np.arange(1, variances.shape[0] + 1)

        fig, ax = plt.subplots(1)
        ax.plot(x, variances, '-xk')
        if smooth:
            ax.set(
                xlabel='Iteration #',
                ylabel='Mean variance')
        else:
            ax.set(
                xlabel='Model #',
                ylabel='Variance')
        plt.savefig(path, bbox_inches='tight')
        plt.close(fig)
