import params as work_parameters
from pytomo.inversion.na import NeighbouhoodAlgorithm, InputFile
from pytomo.inversion.inversionresult import InversionResult
from pytomo import utilities
from pydsm.modelparameters import ModelParameters, ParameterType
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

free_indices = [0, 3]
model_ref, _ = work_parameters.get_model_lininterp(0, 0, 0, 2)
types = [ParameterType.VSH, ParameterType.RADIUS]

figscale = 0.8

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
    out_dir = 'output_' + utilities.get_temporary_str()
    os.mkdir(out_dir)
    points = NeighbouhoodAlgorithm.get_points_for_voronoi(
                result.perturbations,
                result.meta['range_dict'],
                types)
    points = points[:, free_indices]
    misfits = result.misfit_dict['variance'].mean(axis=1)
    n_r = result.meta['n_r']
    n_s = result.meta['n_s']
    n_mod = points.shape[0]
    i_out = 0
    for imod in range(n_mod):
        figpath = os.path.join(
            out_dir, 'voronoi_{:05d}.png'.format(imod))
        fig = plt.figure(figsize=(13*figscale, 5*figscale))
        gs = gridspec.GridSpec(1, 3, width_ratios=[2,1,1])
        # plot voronoi cells
        ax0 = fig.add_subplot(gs[0])
        _, _, colormap = NeighbouhoodAlgorithm.plot_voronoi_2d(
            points[:imod+1], misfits[:imod+1],
            xlim=[-0.5,0.5], ylim=[-0.5,0.5], ax=ax0)
        ax0.set(
            xlabel='dV (km/s)',
            ylabel='dH (km)')
        ax0.set_yticklabels(
            ['{:.0f}'.format(v*380.) for v in ax0.get_yticks()])
        fig.colorbar(
            colormap, ax=ax0, label='Waveform misfit',
            shrink=0.5, fraction=0.07, pad=0.2,
            orientation='horizontal')
        # plot model
        ax1 = fig.add_subplot(gs[1])
        result.plot_models(
            types=types, n_best=1, n_mod=imod+1, ax=ax1,
            color='black', label='best model',
            linewidth=2.)
        work_parameters.get_model_syntest2().plot(
            types=types, ax=ax1,
            color='cyan', label='target',
            linewidth=2.)
        model_ref.plot(
            types=types, ax=ax1,
            color='gray', label='ref',
            linestyle='dashed')
        ax1.set(
            ylim=[3480, 4000],
            xlim=[6.5, 8.])
        ax1.legend(loc='upper right')
        pos1 = list(ax1.get_position().bounds)
        pos1[0] -= 0.04 * figscale
        ax1.set_position(pos1)
        # plot waveforms
        ax2 = fig.add_subplot(gs[2])
        if (i_out+1 < len(indices_better)
            and indices_better[i_out+1] == imod):
            i_out += 1
        result.plot_event(outputs[i_out], 0, ax2)
        pos2 = list(ax2.get_position().bounds)
        pos2[0] -= 0.03 * figscale
        ax2.set_position(pos2)
        ax2.set_title('Transverse')

        fig.suptitle('Model #{}'.format(imod))
        plt.savefig(
            figpath, bbox_inches='tight', dpi=250)
        plt.close(fig)
