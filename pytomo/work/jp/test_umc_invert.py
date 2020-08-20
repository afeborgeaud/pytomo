from pytomo.inversion.cmc import ConstrainedMonteCarlo
from pytomo.work.jp.params import get_model, get_dataset
from pydsm.seismicmodel import SeismicModel
from pydsm.modelparameters import ModelParameters, ParameterType
import numpy as np
import matplotlib.pyplot as plt
from pydsm.event import Event
from pydsm.station import Station
from pydsm.utils.cmtcatalog import read_catalog
from pydsm.dataset import Dataset
from pydsm.dsm import PyDSMInput, compute, compute_models_parallel
from pydsm.windowmaker import WindowMaker
from pydsm.component import Component
from mpi4py import MPI
import matplotlib.colors as colors
import matplotlib.cm as cmx


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    n_cores = comm.Get_size()
    rank = comm.Get_rank()
    
    if rank == 0:
        n_upper_mantle = 20
        n_mtz = 10
        n_lower_mantle = 12
        types = [ParameterType.VSH]
        g = 1.
        l = 1.
        seed = 42

        n_samples = 2

        model_ref, model_params = get_model(
            n_upper_mantle, n_mtz, n_lower_mantle, types)

        cov = ConstrainedMonteCarlo.smooth_damp_cov(
            model_params._n_nodes, g, l)

        cmc = ConstrainedMonteCarlo(
            model_ref, model_params, cov,
            mesh_type='boxcar', seed=seed)
        sample_models = cmc.sample_models(n_samples)
    else:
        model_params = None
        sample_models = None
        model_ref = None

    tlen = 1638.4
    nspc = 64
    sampling_hz = 20

    dataset = get_dataset(tlen, nspc, sampling_hz)

    windows_S = WindowMaker.windows_from_dataset(
        dataset, 'prem', ['S'],
        [Component.T], t_before=30., t_after=50.)
    windows_P = WindowMaker.windows_from_dataset(
        dataset, 'prem', ['P'],
        [Component.Z], t_before=30., t_after=50.)
    windows = windows_S + windows_P

    outputs = compute_models_parallel(
        dataset, sample_models, tlen, nspc, sampling_hz,
        comm, mode=0)

    if rank == 0:
        misfit_dict = cmc.process_outputs(
            outputs, dataset, sample_models, windows)
        print(misfit_dict)
        avg_corrs = misfit_dict['corr'].mean(axis=1)
        print(avg_corrs)

        fig, ax = dataset.plot_event(
            0, windows, component=Component.Z,
            align_zero=True, color='black')
        cycler = plt.rcParams['axes.prop_cycle']
        for imod, sty in enumerate(cycler[:n_samples]):
            _, ax = outputs[imod][0].plot_component(
                Component.Z, windows, ax=ax, align_zero=True, **sty)
        plt.show()

        cm = plt.get_cmap('Greys_r')
        c_norm  = colors.Normalize(vmin=avg_corrs.min(),
                                   vmax=avg_corrs.max()*1.1)
        scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=cm)

        color_val = scalar_map.to_rgba(avg_corrs[0])
        fig, ax = sample_models[0].plot(
            types=[ParameterType.VPV], color=color_val)
        for i, sample in enumerate(sample_models[1:]):
            color_val = scalar_map.to_rgba(avg_corrs[i])
            sample.plot(ax=ax, types=[ParameterType.VPV], color=color_val)
        model_ref.plot(ax=ax, types=[ParameterType.VPV], color='red')
        ax.get_legend().remove()
        #ax.set(ylim=[model_params._radii[0]-100, 6371], xlim=[3, 7])
        ax.set(ylim=[model_params._radii[0]-100, 6371])
        plt.show()