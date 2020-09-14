from pytomo.inversion.cmc import ConstrainedMonteCarlo, InputFile
from pytomo.work.jp import params as work_parameters
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
    mode = params['mode']
    freq = params['freq']
    freq2 = params['freq2']
    filter_type = params['filter_type']
    # sac_files = params['sac_files']
    distance_min = params['distance_min']
    distance_max = params['distance_max']
    t_before = params['t_before']
    t_after = params['t_after']
    # stf_catalog = params['stf_catalog']

    n_distinct_comp_phase = 1

    n_pass = n_mod // n_block
    assert n_mod % n_block == 0
    if verbose > 1:
        print('n_pass={}'.format(n_pass))
    
    if rank == 0:
        n_upper_mantle = 20 #20
        n_mtz = 10 #10
        n_lower_mantle = 12 #12
        types = [ParameterType.VSH]
        g = 0.
        l = 1.
        seed = 42

        model_ref, model_params = work_parameters.get_model(
            n_upper_mantle, n_mtz, n_lower_mantle, types, verbose=verbose)

        cov = ConstrainedMonteCarlo.smooth_damp_cov(
            model_params._n_grd_params, g, l)

        cmc = ConstrainedMonteCarlo(
            model_ref, model_params, cov,
            mesh_type='boxcar', seed=seed)
        models = cmc.sample_models(n_mod)
    else:
        model_params = None
        models = None
        model_ref = None
    
    if rank == 0:
        dataset, output_ref = work_parameters.get_dataset_syntest1(
            tlen, nspc, sampling_hz, mode=mode)
        # dataset = work_parameters.get_dataset(
        #   tlen, nspc, sampling_hz, mode=mode)
        # dataset = Dataset.dataset_from_sac(sac_files, headonly=False)
        if filter_type is not None:
            dataset.filter(freq, freq2, filter_type)

        windows_s = WindowMaker.windows_from_dataset(
            dataset, 'prem', ['S'],
            [Component.T], t_before=t_before, t_after=t_after)
        # windows_p = WindowMaker.windows_from_dataset(
        #     dataset, 'prem', ['p', 'P'],
        #     [Component.Z], t_before=30., t_after=50.)
        windows = windows_s #+ windows_p
        windows = [
            w for w in windows
            if w.get_epicentral_distance() >= distance_min
            and w.get_epicentral_distance() <= distance_max]

        npts_max = int((t_before+t_after) * sampling_hz)
        dataset.apply_windows(
            windows, n_distinct_comp_phase, npts_max, buffer=0.)
    else:
        windows = None
        dataset = None
    dataset = comm.bcast(dataset, root=0)

    ipass = 0
    result = InversionResult(
                dataset, windows)
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
            comm, mode=mode, verbose=verbose)
        
        if rank == 0:
            if filter_type is not None:
                for imod in range(len(outputs)):
                    for iev in range(len(outputs[0])):
                        outputs[imod][iev].filter(
                            freq, freq2, filter_type)

            misfit_dict = cmc.process_outputs(
                outputs, dataset, current_models, windows)
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

        fig, ax = dataset.plot_event(
            0, windows, component=Component.T,
            align_zero=True, color='black')
        cycler = plt.rcParams['axes.prop_cycle']
        for imod, sty in enumerate(cycler[:n_block]):
            _, ax = outputs[imod][0].plot_component(
                Component.T, windows, ax=ax, align_zero=True, **sty)
        plt.savefig('profiles_syntest2.pdf')
        plt.close(fig)
        # plt.show()

        fig, ax = result.plot_models(types=[ParameterType.VSH], n_best=3)
        # fig, ax = work_parameters.get_model_syntest1().plot(
        #     types=[ParameterType.VSH], ax=ax, color='red')
        fig, ax = work_parameters.get_model_syntest2().plot(
            types=[ParameterType.VSH], ax=ax, color='red')
        fig, ax = SeismicModel.ak135().plot(
            types=[ParameterType.VSH], ax=ax, color='blue')
        ax.set(ylim=[model_params._radii[0]-100, 6371])
        ax.get_legend().remove()
        plt.savefig('recovered_models_syntest2.pdf')
        # plt.show()
        plt.close(fig)