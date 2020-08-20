from pytomo.inversion.cmc import ConstrainedMonteCarlo, InputFile
from pytomo.work.jp.params import get_model, get_dataset
from pytomo.inversion.inversionresult import InversionResult
from pydsm.seismicmodel import SeismicModel
from pydsm.modelparameters import ModelParameters, ParameterType
import numpy as np
import time
import matplotlib.pyplot as plt
from pydsm.event import Event
from pydsm.station import Station
from pydsm.utils.cmtcatalog import read_catalog
from pydsm.dataset import Dataset
from pydsm.dsm import PyDSMInput, compute, compute_models_parallel
from pydsm.windowmaker import WindowMaker
from pydsm.component import Component
from mpi4py import MPI
import sys

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    n_cores = comm.Get_size()
    rank = comm.Get_rank()

    input_file = InputFile(sys.argv[1])
    params = input_file.read()

    tlen = params['tlen']
    nspc = params['nspc']
    sampling_hz = params['sampling_hz']
    verbose = params['verbose']
    result_path = params['result_path']
    n_mod = params['n_mod']
    n_block = params['n_block']
    
    n_pass = n_mod // n_block
    assert n_mod % n_block == 0
    
    if rank == 0:
        n_upper_mantle = 20
        n_mtz = 10
        n_lower_mantle = 12
        types = [ParameterType.VSH]
        g = 1.
        l = 1.
        seed = 42

        model_ref, model_params = get_model(
            n_upper_mantle, n_mtz, n_lower_mantle, types, verbose=verbose)

        cov = ConstrainedMonteCarlo.smooth_damp_cov(
            model_params._n_nodes, g, l)

        cmc = ConstrainedMonteCarlo(
            model_ref, model_params, cov,
            mesh_type='boxcar', seed=seed)
        models = cmc.sample_models(n_mod)
    else:
        model_params = None
        models = None
        model_ref = None

    dataset = get_dataset(tlen, nspc, sampling_hz)
    
    if rank == 0:
        windows_S = WindowMaker.windows_from_dataset(
            dataset, 'prem', ['S'],
            [Component.T], t_before=30., t_after=50.)
        windows_P = WindowMaker.windows_from_dataset(
            dataset, 'prem', ['P'],
            [Component.Z], t_before=30., t_after=50.)
        windows = windows_S + windows_P
    else:
        windows = None

    ipass = 0
    misfit_dict = None
    if rank == 0:
        start_time = time.time_ns()
    while ipass < n_pass:
        if rank == 0:
            start = ipass * n_block
            end = start + n_block
            current_models = models[start:end]
        else:
            current_models = None
        outputs = compute_models_parallel(
            dataset, current_models, tlen, nspc, sampling_hz,
            comm, mode=0)
        
        if rank == 0:
            if misfit_dict is None:
                misfit_dict = cmc.process_outputs(
                    outputs, dataset, current_models, windows)
                result = InversionResult(
                    dataset, current_models, windows, misfit_dict)
            else:
                result.add_result(current_models, misfit_dict)

        ipass += 1
        comm.Barrier()

    if rank == 0:
        end_time = time.time_ns()
        result.save(result_path)
        if verbose > 0:
            print('Models and misfits computation done in {} s'
                .format((end_time-start_time) * 1e-9))
            print('Results saved to \'{}\''.format(result_path))

    # if rank == 0:
    #     fig, ax = result.plot_models(types=[ParameterType.VSH])
    #     ax.set(ylim=[model_params._radii[0]-100, 6371])
    #     plt.show()