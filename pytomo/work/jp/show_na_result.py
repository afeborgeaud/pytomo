import pytomo.work.jp.params as work_parameters
from pytomo.inversion.na import NeighbouhoodAlgorithm, InputFile
from pytomo.inversion.inversionresult import InversionResult
from pytomo import utilities
from pydsm.modelparameters import ModelParameters, ParameterType
from pydsm.seismicmodel import SeismicModel
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from mpi4py import MPI
import os
import sys

comm = MPI.COMM_WORLD
n_core = comm.Get_size()
rank = comm.Get_rank()

result = InversionResult.load(sys.argv[1])
loaded = True

model_ref = SeismicModel.ak135()
types = [ParameterType.VSH, ParameterType.RADIUS]

figscale = 1.

# compute best models
if rank == 0:
    indices_better = [0]
    for imod in range(1, result.meta['n_mod']):
        i_best = result.get_indices_best_models(n_best=1, n_mod=imod+1)
        if i_best != indices_better[-1]:
            indices_better.append(imod)
    models = [result.models[i] for i in indices_better]
else:
    models = None
outputs = result.compute_models(models, comm)

# plot results
if rank == 0:
    out_dir = 'figures_' + utilities.get_temporary_str()
    os.mkdir(out_dir)
    misfits = result.misfit_dict['variance'].mean(axis=1)
    n_r = result.meta['n_r']
    n_s = result.meta['n_s']
    n_mod = len(models)
    i_out = 0
    for imod in range(len(models)):
        figpath = os.path.join(
            out_dir, 'result_{:05d}.png'.format(imod))
        fig = plt.figure(figsize=(10*figscale, 5*figscale))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1,1,1])
        # plot model
        ax1 = fig.add_subplot(gs[0])
        models[imod].plot(
            types=types, ax=ax1,
            color='black', label='best model',
            linewidth=1.)
        work_parameters.get_model_syntest3().plot(
            types=types, ax=ax1,
            color='red', label='target',
            linewidth=1.)
        model_ref.plot(
            types=types, ax=ax1,
            color='gray', label='ref',
            linestyle='dashed')
        ax1.set(
            ylim=[5271, 6371],
            xlim=[3.3, 6.8])
        ax1.legend(loc='upper right')
        pos1 = list(ax1.get_position().bounds)
        pos1[0] -= 0.04 * figscale
        ax1.set_position(pos1)
        # plot waveforms
        ax2 = fig.add_subplot(gs[1])
        result.plot_event(
            outputs[imod], 0, ax2, color='red',
            linewidth=0.5)
        pos2 = list(ax2.get_position().bounds)
        pos2[0] -= 0.03 * figscale
        ax2.set_position(pos2)
        ax2.set_title('Transverse')

        fig.suptitle('Model #{}'.format(imod))
        plt.savefig(
            figpath, bbox_inches='tight', dpi=250)
        plt.close(fig)
