import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import pickle

class InversionResult:
    '''Pack the inversion results, models and dataset needed to
    reproduce the results

    Args:
        dataset (pydsm.dataset): dataset
        models (list(pydsm.seismicmodel)): seismic models. The order
            is the same as for axis 0 of misfit_dict.
        windows (list(pydsm.window)): time windows. The order
            is the same as for axis 1 of misfit_dict.
        misfit_dict (dict): values are ndarray((n_models, n_windows))
            containing misfit values (corr, variance)
    '''

    def __init__(self, dataset, windows):
        self.dataset = dataset
        self.windows = windows
        self.misfit_dict = dict()
        self.models = []
        self.perturbations = []
    
    def add_result(self, models, misfit_dict, perturbations):
        '''Add misfit_dict for new models to current inversion result.
        The dataset and windows must be the same.
        
        Args:
            models (list(pydsm.seismicmodel)): seismic models. The order
            is the same as for axis 0 of misfit_dict.
            misfit_dict (dict): values are 
                ndarray((n_models, n_windows)) containing misfit
                values (corr, variance)
        '''
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
    
    # TODO delete
    # def get_model_perturbations(
    #         self, model_ref, types, smooth=True, n_s=None,
    #         in_percent=False):
    def get_model_perturbations(
            self, smooth=True, n_s=None,
            in_percent=False):
        '''Get the model perturbations w.r.t. model_ref to use as
        a convergence criteria
        Args:
            model_ref (pydsm.SeismicModel): reference model
            types (list(pydsm.ParameterType)): parameter types
                (e.g., ParameterType.VSH)
            smooth (bool): smooth over n_s (True)
            n_s (int): number of models computed at each iteration
            in_percent (bool): returns perturbations as percent (False)
        Returns:
            perturbations (ndarray): array of perturbations.
                If smooth, shape=(n_iteration,), else shape=(n_models,)
        '''
        # TODO delete
        # pert_list = []
        # for i in range(len(self.models)):
        #     per_arr = self.models[i].get_perturbations_to(
        #         model_ref, types, in_percent)
        #     pert_list.append(per_arr)
        
        # perturbations = np.array(pert_list)

        perturbations = np.array(self.perturbations)

        if smooth:
            n_it = int(len(self.models)//n_s)
            if len(self.models) % n_s != 0:
                n_it += 1
            perturbations_ = np.zeros((n_it, perturbations.shape[1]))
            for i in range(n_it):
                s = i * n_s
                e = s + n_s
                perturbations_[i] = perturbations[s:e].mean(axis=0)
            perturbations = perturbations_

            # if len(self.models) % n_s != 0:
            #     n = int(len(self.models)//n_s * n_s + n_s)
            #     perturbations = np.pad(
            #         perturbations, (0,n), 'constant',
            #         constant_values=(0,0))
            # perturbations = perturbations.reshape(
            #     (n_s, -1, perturbations.shape[1])).mean(axis=0)

        return perturbations

    def get_variances(self, smooth=True, n_s=None):
        variances = self.misfit_dict['variance'].mean(axis=1)
        if smooth:
            if len(variances) % n_s != 0:
                n = int(len(variances)//n_s * n_s + n_s)
                perturbations = np.pad(
                    perturbations, (0,n), 'constant',
                    constant_values=(0,0))
            variances = variances.reshape((-1, n_s)).mean(axis=1)
        return variances
    
    def save(self, path):
        '''Save self using pickle.dump().
        Args:
            path (str): name of the output file
        '''
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        '''Read file into self using pickle.load().
        Args:
            path (str): name of the file that contains self
        '''
        with open(path, 'rb') as f:
            output = pickle.load(f)
        return output
    
    def plot_models(self, types, n_best=-1, **kwargs):
        '''Plot models colored by misfit value.

        Args:
            types (list(pydsm.modelparameter.ParameterType)):
                types e.g., RHO
        '''
        avg_corrs = self.misfit_dict['corr'].mean(axis=1)
        avg_vars = self.misfit_dict['variance'].mean(axis=1)
        indices_best = np.arange(len(avg_corrs), dtype=int)
        if type(n_best)==int and n_best > 0:
            n_best = min(n_best, len(avg_corrs))
            indices_best = avg_corrs.argsort()[:n_best]
            print(avg_corrs[indices_best])
            print(avg_corrs.max())
            print(avg_vars[indices_best])
            print(avg_vars.max())
            print(indices_best)
        elif type(n_best)==float:
            if n_best <= 1:
                # TODO make it a percentage of the number of models
                indices_best = np.where(avg_corrs <= (1+n_best)*avg_corrs.min())

        cm = plt.get_cmap('Greys_r')
        c_norm  = colors.Normalize(vmin=avg_corrs.min(),
                                   vmax=avg_corrs.max()*1.1)
        scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=cm)

        if 'color' not in kwargs:
            color = scalar_map.to_rgba(avg_corrs[indices_best[0]])
        else:
            color = kwargs['color']
            kwargs.pop('color', None)
            
        fig, ax = self.models[indices_best[0]].plot(
            types=types, color=color, **kwargs)

        for i in indices_best[1:]:
            color = scalar_map.to_rgba(avg_corrs[i])
            self.models[i].plot(ax=ax, types=types, color=color, **kwargs)

        # model_ref.plot(ax=ax, types=[ParameterType.VPV], color='red')
        ax.get_legend().remove()
        
        return fig, ax
