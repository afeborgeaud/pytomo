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

    def __init__(self, dataset, models, windows, misfit_dict):
        self.dataset = dataset
        self.models = models
        self.windows = windows
        self.misfit_dict = misfit_dict
    
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
        self.models += models
        for misfit_name in self.misfit_dict.keys():
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
    
    def plot_models(self, types, **kwargs):
        '''Plot models colored by misfit value.

        Args:
            types (list(pydsm.modelparameter.ParameterType)):
                types e.g., RHO
        '''
        avg_corrs = self.misfit_dict['corr'].mean(axis=1)

        cm = plt.get_cmap('Greys_r')
        c_norm  = colors.Normalize(vmin=avg_corrs.min(),
                                   vmax=avg_corrs.max()*1.3)
        scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=cm)

        color_val = scalar_map.to_rgba(avg_corrs[0])
        fig, ax = self.models[0].plot(
            types=types, color=color_val, **kwargs)
        for i, sample in enumerate(self.models[1:]):
            color_val = scalar_map.to_rgba(avg_corrs[i])
            sample.plot(ax=ax, types=types, color=color_val, **kwargs)
        # model_ref.plot(ax=ax, types=[ParameterType.VPV], color='red')
        ax.get_legend().remove()
        
        return fig, ax
