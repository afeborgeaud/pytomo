from pytomo.inversion.cmc import ConstrainedMonteCarlo, InputFile
from pytomo.work.jp.params import get_model, get_dataset
from pytomo.inversion.inversionresult import InversionResult
from dsmpy.seismicmodel import SeismicModel
from dsmpy.modelparameters import ModelParameters, ParameterType
import numpy as np
import time
import matplotlib.pyplot as plt
from dsmpy.event import Event
from dsmpy.station import Station
from dsmpy.utils.cmtcatalog import read_catalog
from dsmpy.dataset import Dataset
from dsmpy.dsm import PyDSMInput, compute, compute_models_parallel
from dsmpy.windowmaker import WindowMaker
from dsmpy.component import Component
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
    mode = params['mode']
    freq = params['freq']
    freq2 = params['freq2']
    filter_type = params['filter_type']

    n_pass = n_mod // n_block
    assert n_mod % n_block == 0
    if verbose > 1:
        print('n_pass={}'.format(n_pass))
    
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

    dataset = get_dataset(tlen, nspc, sampling_hz, mode=mode)

    if filter_type is not None:
        dataset.filter(freq, freq2, filter_type)
    
    if rank == 0:
        windows_S = WindowMaker.windows_from_dataset(
            dataset, 'prem', ['s', 'S'],
            [Component.T], t_before=30., t_after=50.)
        windows_P = WindowMaker.windows_from_dataset(
            dataset, 'prem', ['p', 'P'],
            [Component.Z], t_before=30., t_after=50.)
        windows = windows_S #+ windows_P
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
            comm, mode=mode)
        
        if rank == 0:
            if filter_type is not None:
                for imod in range(len(outputs)):
                    for iev in range(len(outputs[0])):
                        outputs[imod][iev].filter(
                            freq, freq2, filter_type)

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

        # fig, ax = dataset.plot_event(
        #     0, windows, component=Component.T,
        #     align_zero=True, color='black')
        # cycler = plt.rcParams['axes.prop_cycle']
        # for imod, sty in enumerate(cycler[:n_block]):
        #     _, ax = outputs[imod][0].plot_component(
        #         Component.T, windows, ax=ax, align_zero=True, **sty)
        # plt.show()

        # fig, ax = result.plot_models(types=[ParameterType.VSH])
        # ax.set(ylim=[model_params._radii[0]-100, 6371])
        # plt.show()