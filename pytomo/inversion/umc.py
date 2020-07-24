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

    def sample_one_model(self):
        value_dict = dict()
        for param_type in self.model_params._types:
            values = self.rng.uniform(
                self.range_dict[param_type][0],
                self.range_dict[param_type][1],
                self.model_params._n_nodes-1)
            value_dict[param_type] = values
        model_sample = self.model.build_model(
            self.mesh, self.model_params, value_dict)
        return model_sample

    def sample_models(self, ns):
        models = [self.sample_one_model() for i in range(ns)]
        return models

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

    fig, ax = sample_models[0].plot(parameters=['vsh', 'vph'])
    for sample in sample_models[1:]:
        sample.plot(ax=ax, parameters=['vsh', 'vph'])
    ax.get_legend().remove()
    plt.show()
