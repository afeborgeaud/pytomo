import params as work_parameters
from pytomo.inversion.na import NeighbouhoodAlgorithm, InputFile
from pytomo.inversion.inversionresult import InversionResult
from dsmpy.modelparameters import ModelParameters, ParameterType
from dsmpy.dataset import Dataset
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI
import os
import sys
import glob

comm = MPI.COMM_WORLD
n_core = comm.Get_size()
rank = comm.Get_rank()

input_file = sys.argv[1]

# Set the SAC file paths
sac_path = "/mnt/ntfs/anselme/work/japan/DATA/2*/*T"
sac_files = list(glob.iglob(sac_path))

# Create the dataset
dataset = Dataset.dataset_from_sac(sac_files, headonly=False)

# Set the model parameters
types = [ParameterType.VSH, ParameterType.RADIUS]
model_ref, model_params = work_parameters.get_model_lininterp(
    types=types, verbose=0, discontinuous=True)

# Set the constraints to model parameters
mask_dict = dict()
mask_dict[ParameterType.VSH] = np.ones(
    model_params._n_grd_params, dtype='bool')
mask_dict[ParameterType.RADIUS] = np.zeros(
    model_params._n_grd_params, dtype='bool')
mask_dict[ParameterType.VSH][[0, 1]] = False
mask_dict[ParameterType.VSH][[-1, -2]] = False
mask_dict[ParameterType.RADIUS][[i for i in range(4, 10)]] = True

discon_arr = np.zeros(
    model_params._n_nodes, dtype='bool')
discon_arr[2] = True
discon_arr[4] = True

model_params.set_constraints(
    mask_dict=mask_dict, discon_arr=discon_arr)

# Set the model parameter ranges
range_dict = dict()
for param_type in model_params._types:
    range_arr = np.zeros((model_params._n_grd_params, 2), dtype='float')
    if param_type == ParameterType.RADIUS:
        range_arr[4, 0] = -40.
        range_arr[4, 1] = 40.
        range_arr[5, 0] = -40.
        range_arr[5, 1] = 40.
        range_arr[6, 0] = -65.  # (250-2*40)/2 - 20
        range_arr[6, 1] = 65.
        range_arr[7, 0] = -65.
        range_arr[7, 1] = 65.
        range_arr[8, 0] = -40.
        range_arr[8, 1] = 40.
        range_arr[9, 0] = -40.
        range_arr[9, 1] = 40.
    if param_type == ParameterType.VSH:
        range_arr[:, 0] = -0.3
        range_arr[:, 1] = 0.3
    range_dict[param_type] = range_arr

# create NeighbouhoodAlgorithm object
na = NeighbouhoodAlgorithm.from_file(
    input_file, model_ref, model_params, range_dict,
    dataset, comm)

# run NA
log_path = os.path.join(
    na.out_dir, 'log_{}'.format(rank))
log = open(log_path, 'w', buffering=1)

result = na.compute(comm, log)

# plot inverted model
if rank == 0:
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 6))
    result.plot_models(
        types=types, n_best=1, ax=ax,
        color='black', label='best model')
    na.model_ref.plot(
        types=types, ax=ax,
        color='gray', label='ak135')
    ax.set(
        ylim=[5271, 6371],
        xlim=[3.3, 6.8])
    ax.legend()
    fig_path = os.path.join(
        na.out_dir, 'inverted_models.pdf')
    plt.savefig(
        fig_path,
        bbox_inches='tight')
    plt.close(fig)

log.close()
