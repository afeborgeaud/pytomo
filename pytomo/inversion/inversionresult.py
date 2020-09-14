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
    
    def add_result(self, models, misfit_dict):
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
        for misfit_name in misfit_dict.keys():
            if misfit_name not in self.misfit_dict:
                self.misfit_dict[misfit_name] = misfit_dict[misfit_name]
            else:
                self.misfit_dict[misfit_name] = np.vstack(
                    (self.misfit_dict[misfit_name],
                    misfit_dict[misfit_name]))
    
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
            types=types, color='red', **kwargs)

        for i in indices_best[1:]:
            color = scalar_map.to_rgba(avg_corrs[i])
            self.models[i].plot(ax=ax, types=types, color='red', **kwargs)

        # model_ref.plot(ax=ax, types=[ParameterType.VPV], color='red')
        ax.get_legend().remove()
        
        return fig, ax
