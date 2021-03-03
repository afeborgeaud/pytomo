from pytomo.inversion.inversionresult import InversionResult
from pytomo import utilities
from dsmpy.modelparameters import ModelParameters, ParameterType
from dsmpy.seismicmodel import SeismicModel
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

# compute best model
if rank == 0:
    i_best = result.get_indices_best_models(n_best=-1)[1]
    model_best = result.models[i_best]
    model_best._model_id = 'mod{}'.format(i_best)
    models = [model_ref, model_best]
else:
    models = None
outputs = result.compute_models(models, comm)

# plot results
if rank == 0:
    out_dir = 'figures_' + utilities.get_temporary_str()
    os.mkdir(out_dir)
    for imod in range(len(models)):
        for iev, event in enumerate(result.dataset.events):
            figpath = os.path.join(
                out_dir, '{}_{}.pdf'.format(event, models[imod]._model_id))
            fig = plt.figure(figsize=(6, 10))
            gs = gridspec.GridSpec(1, 1, width_ratios=[1])
            # plot waveforms
            ax0 = fig.add_subplot(gs[0])
            result.plot_event(
                outputs[0], iev, ax0, color='red',
                linewidth=0.5)
            ax0.set_title('Transverse')

            plt.savefig(
                figpath, bbox_inches='tight')
            plt.close(fig)
