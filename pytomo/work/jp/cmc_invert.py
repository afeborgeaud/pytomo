from pytomo.inversion.cmcutils import ConstrainedMonteCarloUtils, InputFile
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

class ConstrainedMonteCarlo:
    '''Inversion using constrained Monte Carlo
    '''
    def __init__(
            self, tlen, nspc, sampling_hz, mode, n_mod, n_block,
            phases, components, t_before, t_after, filter_type,
            freq, freq2, sac_files, distance_min, distance_max,
            stf_catalog, result_path, seed, verbose, comm):
        self.tlen = tlen
        self.nspc = nspc
        self.sampling_hz = sampling_hz
        self.mode = mode
        self.n_mod = n_mod
        self.n_block = n_block
        self.phases = phases
        self.components = components
        self.t_before = t_before
        self.t_after = t_after
        self.filter_type = filter_type
        self.freq = freq
        self.freq2 = freq2
        self.sac_files = sac_files
        self.distance_min = distance_min
        self.distance_max = distance_max
        self.stf_catalog = stf_catalog
        self.result_path = result_path
        self.seed = seed
        self.verbose = verbose
        self.comm = comm

    @classmethod
    def from_file(cls, input_file_path, comm):
        '''Build object from the input file
        Args:
            input_file_path (str): path of the input file
            comm (mpi4py.Comm_world): MPI communicator
        Returns:
            ConstrainedMonteCarlo object
        '''
        params = InputFile(input_file_path).read()

        tlen = params['tlen']
        nspc = params['nspc']
        sampling_hz = params['sampling_hz']
        mode = params['mode']
        n_mod = params['n_mod']
        n_block = params['n_block']
        phases = params['phases']
        components = params['components']
        t_before = params['t_before']
        t_after = params['t_after']
        filter_type = params['filter_type']
        freq = params['freq']
        freq2 = params['freq2']
        sac_files = params['sac_files']
        distance_min = params['distance_min']
        distance_max = params['distance_max']
        stf_catalog = params['stf_catalog']
        result_path = params['result_path']
        seed = params['seed']
        verbose = params['verbose']

        return cls(
            tlen, nspc, sampling_hz, mode, n_mod, n_block,
            phases, components, t_before, t_after, filter_type,
            freq, freq2, sac_files, distance_min, distance_max,
            stf_catalog, result_path, seed, verbose, comm)

    def get_windows(self, dataset):
        windows = []
        for i in len(self.phases):
            windows_tmp = WindowMaker.windows_from_dataset(
                dataset, 'ak135', [self.phases[i]],
                [self.components[i]],
                t_before=self.t_before, t_after=self.t_after)
            windows += windows_tmp
        windows = [
            w for w in windows
            if w.get_epicentral_distance() >= self.distance_min
            and w.get_epicentral_distance() <= self.distance_max]
        return windows

    def compute(self, comm):
        '''Run the inversion.
        Args:
            comm (mpi4py.Comm_world): MPI Communicator
        Returns:
            result (InversionResult): inversion results
        '''
        rank = comm.Get_rank()

        n_distinct_comp_phase = len(self.phases)

        n_pass = self.n_mod // self.n_block
        assert self.n_mod % self.n_block == 0
        if self.verbose > 1:
            print('n_pass={}'.format(n_pass))
        
        if rank == 0:
            n_upper_mantle = 20 #20
            n_mtz = 10 #10
            n_lower_mantle = 12 #12
            types = [ParameterType.VSH]
            g = 0.
            l = 1.

            model_ref, model_params = work_parameters.get_model(
                n_upper_mantle, n_mtz, n_lower_mantle, types,
                verbose=self.verbose)

            cov = ConstrainedMonteCarloUtils.smooth_damp_cov(
                model_params._n_grd_params, g, l)

            cmc = ConstrainedMonteCarloUtils(
                model_ref, model_params, cov,
                mesh_type='boxcar', seed=self.seed)
            models = cmc.sample_models(self.n_mod)
        else:
            model_params = None
            models = None
            model_ref = None
        
        if rank == 0:
            dataset, output_ref = work_parameters.get_dataset_syntest1(
                self.tlen, self.nspc, self.sampling_hz, mode=self.mode)
            # dataset = work_parameters.get_dataset(
            #   tlen, nspc, sampling_hz, mode=mode)
            # dataset = Dataset.dataset_from_sac(sac_files, headonly=False)
            if self.filter_type is not None:
                dataset.filter(self.freq, self.freq2, self.filter_type)

            windows = self.get_windows(dataset)

            npts_max = int((self.t_before+self.t_after) * self.sampling_hz)
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
                start = ipass * self.n_block
                end = start + self.n_block
                current_models = models[start:end]
            else:
                current_models = None
            outputs = compute_models_parallel(
                dataset, current_models, self.tlen, self.nspc,
                self.sampling_hz, self.comm, mode=self.mode,
                verbose=self.verbose)
            
            if rank == 0:
                if self.filter_type is not None:
                    for imod in range(len(outputs)):
                        for iev in range(len(outputs[0])):
                            outputs[imod][iev].filter(
                                self.freq, self.freq2, self.filter_type)

                misfit_dict = process_outputs(
                    outputs, dataset, current_models, windows)
                result.add_result(current_models, misfit_dict)

            ipass += 1
            comm.Barrier()

        if rank == 0:
            end_time = time.time_ns()
            result.save(self.result_path)
            if self.verbose > 0:
                print('Models and misfits computation done in {} s'
                    .format((end_time-start_time) * 1e-9))
                print('Results saved to \'{}\''.format(self.result_path))

        return result  

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    n_cores = comm.Get_size()
    rank = comm.Get_rank()

    input_file = InputFile(sys.argv[1])
    cmc = ConstrainedMonteCarlo.from_file(input_file, comm)

    result = cmc.compute(comm)

    fig, ax = result.plot_models(types=[ParameterType.VSH], n_best=3)
    # fig, ax = work_parameters.get_model_syntest1().plot(
    #     types=[ParameterType.VSH], ax=ax, color='red')
    fig, ax = work_parameters.get_model_syntest2().plot(
        types=[ParameterType.VSH], ax=ax, color='red')
    fig, ax = SeismicModel.ak135().plot(
        types=[ParameterType.VSH], ax=ax, color='blue')
    ax.set(ylim=[6371.-1100, 6371.])
    ax.get_legend().remove()
    plt.savefig('recovered_models_syntest2.pdf')
    # plt.show()
    plt.close(fig)
    
    # fig, ax = dataset.plot_event(
    #     0, windows, component=Component.T,
    #     align_zero=True, color='black')
    # cycler = plt.rcParams['axes.prop_cycle']
    # for imod, sty in enumerate(cycler[:n_block]):
    #     _, ax = outputs[imod][0].plot_component(
    #         Component.T, windows, ax=ax, align_zero=True, **sty)
    # plt.savefig('profiles_syntest2.pdf')
    # plt.close(fig)
    # # plt.show()