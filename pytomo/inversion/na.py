from pytomo.inversion.umcutils import UniformMonteCarlo
from pytomo.work.ca import params as work_parameters
from pytomo.inversion.inversionresult import InversionResult
import pytomo.inversion.voronoi as voronoi
from scipy.spatial import voronoi_plot_2d, Voronoi
from pydsm.seismicmodel import SeismicModel
from pydsm.modelparameters import ModelParameters, ParameterType
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from pydsm.event import Event
from pydsm.station import Station
from pydsm.utils.cmtcatalog import read_catalog
from pydsm.dataset import Dataset
from pydsm.dsm import PyDSMInput, compute, compute_models_parallel
from pydsm.windowmaker import WindowMaker
from pydsm.component import Component
from mpi4py import MPI
import sys
import os
import glob
import warnings

class InputFile:
    """Input file for NeighbourAlgorithm (NA) inversion.

    Args:
        input_file (str): path of NA input file
    """
    def __init__(self, input_file):
        self.input_file = input_file
    
    def read(self):
        params = dict()
        params['verbose'] = 0
        params['filter_type'] = None
        params['sac_files'] = None
        params['seed'] = 42
        params['stf_catalog'] = None
        with open(self.input_file, 'r') as f:
            for line in f:
                if line.strip().startswith('#'):
                    continue
                key, value = self._parse_line(line)
                if key is not None:
                    params[key] = value
        if 'phases' in params:
            assert 'components' in params
            assert len(params['phases']) == len(params['components'])
        assert params['n_s'] % params['n_r'] == 0
        assert params['n_mod'] % params['n_s'] == 0
        return params

    def _parse_line(self, line):
        key, value = line.strip().split()[:2]
        if key == 'sac_files':
            full_path = os.path.expanduser(value.strip())
            value_parsed = list(glob.iglob(full_path))
        elif key == 'tlen':
            value_parsed = float(value)
        elif key == 'nspc':
            value_parsed = int(value)
        elif key == 'sampling_hz':
            value_parsed = int(value)
        elif key == 'n_mod':
            value_parsed = int(value)
        elif key == 'n_block':
            value_parsed = int(value)
        elif key == 'n_s':
            value_parsed = int(value)
        elif key == 'n_r':
            value_parsed = int(value)
        elif key == 'mode':
            value_parsed = int(value)
        elif key == 'result_path':
            full_path = os.path.expanduser(value.strip())
            value_parsed = full_path
        elif key == 'verbose':
            value_parsed = int(value)
        elif key == 'freq':
            value_parsed = float(value)
        elif key == 'freq2':
            value_parsed = float(value)
        elif key == 'filter_type':
            value_parsed = value.strip().lower()
        elif key == 'distance_min':
            value_parsed = float(value)
        elif key == 'distance_max':
            value_parsed = float(value)
        elif key == 't_before':
            value_parsed = float(value)
        elif key == 't_after':
            value_parsed = float(value)
        elif key == 'stf_catalog':
            full_path = os.path.expanduser(value.strip())
            value_parsed = full_path
        elif key == 'phases':
            ss = [s.strip() for s in value.strip().split()]
            value_parsed = ss
        elif key == 'components':
            ss = [s.strip() for s in value.strip().split()]
            value_parsed = [Component.parse_component(s) for s in ss]
        elif key == 'seed':
            value_parsed = int(value)
        else:
            print('Warning: key {} undefined. Ignoring.'.format(key))
            return None, None
        return key, value_parsed

class NeighbouhoodAlgorithm:
    '''Implements a neighbourhood algorithm

    Args:
        n_mod: total number of models sampled
        n_s: number of models in one step of the NA (n_mod%n_s=0)
        n_r: number of best-fit models retained in one step of the MA
            (n_r%n_s=0)
    '''
    def __init__(
            self, tlen, nspc, sampling_hz, mode, n_mod, n_block, n_s, n_r,
            phases, components, t_before, t_after, filter_type,
            freq, freq2, sac_files, distance_min, distance_max,
            stf_catalog, result_path, seed, verbose, comm):
        self.tlen = tlen
        self.nspc = nspc
        self.sampling_hz = sampling_hz
        self.mode = mode
        self.n_mod = n_mod
        self.n_block = n_block
        self.n_s = n_s
        self.n_r = n_r
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

        self.rng_gibbs = np.random.default_rng(seed)

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
        n_s = params['n_s']
        n_r = params['n_r']
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
            tlen, nspc, sampling_hz, mode, n_mod, n_block, n_s, n_r,
            phases, components, t_before, t_after, filter_type,
            freq, freq2, sac_files, distance_min, distance_max,
            stf_catalog, result_path, seed, verbose, comm)

    def get_windows(self, dataset):
        windows = []
        for i in range(len(self.phases)):
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

    def filter_outputs(self, outputs):
        if self.filter_type is not None:
            for imod in range(len(outputs)):
                for iev in range(len(outputs[0])):
                    outputs[imod][iev].filter(
                        self.freq, self.freq2, self.filter_type)

    # def get_points_for_voronoi(self, perturbations):
    #     '''
    #     Args:
    #         perturbations (dict): dict of model perturbations
    #     Return:
    #         points: points in a n_grd_params*len(types) dimensional
    #             space to define the voronoi cells
    #     '''
    #     points = np.hstack(
    #         tuple([perturbations[param_type] for param_type in self.types]))

    #     return points

    def get_points_for_voronoi(self, models, model_ref):
        '''
        Args:
            perturbations (dict): dict of model perturbations
        Return:
            points: points in a n_grd_params*len(types) dimensional
                space to define the voronoi cells
        '''
        points = np.zeros(
            (len(models), self.n_grd_param*len(self.types)),
            dtype='float')
        for imod, model in enumerate(models):
            points[imod] = model.get_perturbations_to(
                model_ref, self.types, in_percent=False)

        return points

    def unravel_voronoi_point_bounds(self, point_bounds, imods=None):
        range_dict = dict()
        for itype, param_type in enumerate(self.types):
            start = itype * self.n_grd_param
            end = start + self.n_grd_param
            if imods is None:
                range_dict[param_type] = point_bounds[:, start:end]
            else:
                range_dict[param_type] = point_bounds[imods, start:end]
        return range_dict

    def get_bounds_for_voronoi(self):
        min_bounds = np.zeros(self.n_grd_param*len(self.types), dtype='float')
        max_bounds = np.zeros(self.n_grd_param*len(self.types), dtype='float')
        for itype in range(len(self.types)):
            for igrd in range(self.n_grd_param):
                i = igrd + itype * self.n_grd_param
                min_bounds[i] = self.range_dict[self.types[itype]][igrd, 0]
                max_bounds[i] = self.range_dict[self.types[itype]][igrd, 1]
        return min_bounds, max_bounds

    def compute_one_step(self, umcutils, dataset, models, result, windows, comm):
        #TODO URGENT fix zero output when n_model % n_core != 0
        rank = comm.Get_rank()

        outputs = compute_models_parallel(
            dataset, models, self.tlen, self.nspc,
            self.sampling_hz, self.comm, mode=self.mode,
            verbose=self.verbose)

        if rank == 0:
            self.filter_outputs(outputs)
            misfit_dict = umcutils.process_outputs(
                outputs, dataset, models, windows)
            result.add_result(models, misfit_dict)

    def compute(self, comm, log=None):
        '''Run the inversion.
        Args:
            comm (mpi4py.Comm_world): MPI Communicator
        Returns:
            result (InversionResult): inversion results
        '''
        rank = comm.Get_rank()

        n_distinct_comp_phase = len(self.phases)

        n_pass = self.n_mod // self.n_s
        assert self.n_mod % self.n_s == 0
        if self.verbose > 1:
            print('n_pass={}'.format(n_pass))
        
        self.types = [ParameterType.VSH]

        if rank == 0:
            n_upper_mantle = 0 # 20
            n_mtz = 0 # 10
            n_lower_mantle = 0 # 12
            n_dpp = 3 # 9

            model_ref, model_params = work_parameters.get_model(
                n_upper_mantle, n_mtz, n_lower_mantle, n_dpp, self.types,
                verbose=self.verbose)

            self.n_grd_param = model_params._n_grd_params
            
            range_dict = dict()
            for param_type in self.types:
                range_arr = np.empty((self.n_grd_param, 2), dtype='float')
                range_arr[:, 0] = -0.5
                range_arr[:, 1] = 0.5

                range_dict[param_type] = range_arr

            umcutils = UniformMonteCarlo(
                model_ref, model_params, range_dict,
                mesh_type='boxcar', seed=self.seed)
            models, perturbations = umcutils.sample_models(self.n_s)
        else:
            model_params = None
            models = None
            perturbations = None
            model_ref = None
            umcutils = None
            range_dict = None

        range_dict = comm.bcast(range_dict, root=0)
        self.range_dict = range_dict
        
        if rank == 0:
            # dataset, output_ref = work_parameters.get_dataset_syntest1(
            #     self.tlen, self.nspc, self.sampling_hz, mode=self.mode)
            dataset, output_ref = work_parameters.get_dataset_syntest1(
                self.tlen, self.nspc, self.sampling_hz, mode=self.mode,
                add_noise=False, noise_normalized_std=1.)
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

        result = InversionResult(
                    dataset, windows)

        if rank == 0:
            start_time = time.time_ns()

        # step 0
        self.compute_one_step(
            umcutils, dataset, models, result, windows, comm)
        comm.Barrier()

        # steps 1,...,n_pass-1
        ipass = 1
        while ipass < n_pass:
            if rank == 0:
                # indexing of points corrsespond to that of 
                # perturbations and of that of models
                points = self.get_points_for_voronoi(
                    result.models, model_ref)

                # if points.shape[1] == 2:
                #     if log:
                #         log.write('Building voronoi diag...\n')
                #     start_time = time.time_ns()
                #     vor = Voronoi(points)
                #     end_time = time.time_ns()
                #     if log:
                #         log.write(
                #             'Voronoi diag build in {} s\n'
                #             .format((end_time - start_time)*1e-9))

                if points.shape[1] == 2:
                    misfits = result.misfit_dict['variance'].mean(axis=1)
                    figpath = 'voronoi_syntest1_{}.pdf'.format(ipass)
                    self.save_voronoi_2d(
                        figpath, points, misfits,
                        title='iteration {}'.format(ipass))

                min_bounds, max_bounds = self.get_bounds_for_voronoi()
                indices_best = umcutils.get_best_models(
                    result.misfit_dict, self.n_r)

                models = []
                for imod in range(self.n_r):
                    current_model = result.models[indices_best[imod]]
                    current_perturbations = points[indices_best[imod]]
                    n_step = int(self.n_s//self.n_r)

                    perturbations_arr = np.zeros(
                        (n_step, self.n_grd_param*len(self.types)),
                        dtype='float')
                    perturbations_arr[0] = np.array(current_perturbations)
                    
                    ip = indices_best[imod]
                    # bounds = np.zeros(
                    #     (self.n_grd_param*len(self.types), 2),
                    #     dtype='float')

                    # for idim in range(bounds.shape[0]):

                    current_point = np.array(points[ip])
                    for istep in range(n_step):
                        idim = istep % (self.n_grd_param*len(self.types))
                        itype = int(idim // self.n_grd_param)
                        igrd = idim % self.n_grd_param

                        # calculate bounds
                        start_time = time.time_ns()
                        tmp_bounds1 = voronoi.implicit_find_bound_for_dim(
                            points, points[ip],
                            current_point, idim, n_nearest=120,
                            min_bound=min_bounds[idim],
                            max_bound=max_bounds[idim], step_size=0.001,
                            n_step_max=1000, log=log)
                        tmp_bounds2 = voronoi.implicit_find_bound_for_dim(
                            points, points[ip],
                            current_point, idim, n_nearest=150,
                            min_bound=min_bounds[idim],
                            max_bound=max_bounds[idim], step_size=0.001,
                            n_step_max=1000, log=log)
                        if not np.allclose(tmp_bounds1, tmp_bounds2):
                            print(tmp_bounds1)
                            print(tmp_bounds2)
                            warnings.warn(
                                '''Problems with finding bounds 
                                of Voronoi cell. 
                                Please increase n_nearest''')
                        bounds = tmp_bounds2
                        end_time = time.time_ns()
                        if log:
                            log.write(
                                ('bounds for model {} '
                                + 'for dim {} found in {} s\n')
                                .format(
                                    imod, idim,
                                    (end_time-start_time)*1e-9))

                        lo, up = bounds
                        # per = self.rng_gibbs.uniform(lo, up, 1)[0]
                        per = self.bi_triangle(lo, up)
                        # print(rank, ipass, imod, istep, lo, up, per)

                        value_dict = dict()
                        for param_type in self.types:
                            value_dict[param_type] = np.zeros(
                                self.n_grd_param, dtype='float')
                            if param_type == self.types[itype]:
                                value_dict[param_type][igrd] = per

                        current_model = current_model.build_model(
                            umcutils.mesh, model_params, value_dict)
                        models.append(current_model)
                        
                        if istep > 0:
                            perturbations_arr[istep] = np.array(
                                perturbations_arr[istep-1])
                        perturbations_arr[istep, idim] += per

                        print(
                            '{} {} {} {} {} {} {:.3f} {:.3f} {:.3f}'
                            .format(rank, ipass, imod, istep, idim,
                                    current_point, lo, up, per))
                        current_point[idim] += per

                        # bounds[idim] -= per

                    for param_type in self.types:
                        perturbations_tmp = self.unravel_voronoi_point_bounds(
                            perturbations_arr)
                        perturbations[param_type] = np.vstack((
                            perturbations[param_type],
                            perturbations_tmp[param_type]))
            else:
                models = None

            self.compute_one_step(
                umcutils, dataset, models, result, windows, comm)

            ipass += 1
            comm.Barrier()

        if rank == 0:
            end_time = time.time_ns()
            result.save(self.result_path)
            if self.verbose > 0:
                print('Models and misfits computation done in {} s'
                    .format((end_time-start_time) * 1e-9))
                print('Results saved to \'{}\''.format(self.result_path))

            points = self.get_points_for_voronoi(
                result.models, model_ref)
            if points.shape[1] == 2:
                misfits = result.misfit_dict['variance'].mean(axis=1)
                figpath = 'voronoi_syntest1_{}.pdf'.format(n_pass)
                self.save_voronoi_2d(
                    figpath, points, misfits,
                    title='iteration {}'.format(n_pass))

        return result

    def bi_triangle_cfd_inv(self, x, a, b):
        aa = np.abs(a)
        h = 2. / (aa + b)
        if x < h*aa/4.:
            y = np.sqrt(x*aa/h) - aa
        elif x < h*aa/2.:
            y = -np.sqrt(aa*aa/2. - x*aa/h)
        elif x < (h*aa/2. + h*b/4.):
            y = np.sqrt(x*b/h - aa*b/2.)
        else:
            y = -np.sqrt(b/h * (1-x)) + b
        return y

    def bi_triangle(self, a, b):
        assert (a==b==0) or ((a < b) and (a <= 0) and (b >= 0))
        x_unif = self.rng_gibbs.uniform(0, 1, 1)[0]
        x = self.bi_triangle_cfd_inv(x_unif, a, b)
        return x

    def save_voronoi_2d(self, path, points, misfits, **kwargs):
        '''Save the voronoi diagram to fig path
        Args:
            path (str): path of the output figure
            points (ndarray(npoint,2)): points to build the Voronoi
                diagram
            misfits (ndarray(npoint)): value of misfit for each point
        '''
        # add dummy points
        # stackoverflow.com/questions/20515554/
        # colorize-voronoi-diagram?lq=1
        points_ = np.append(
            points,
            [[999,999], [-999,999], [999,-999], [-999,-999]],
            axis = 0)
        vor = Voronoi(points_)

        # color map
        log_misfits = np.log(misfits)
        cm = plt.get_cmap('hot')
        c_norm  = colors.Normalize(
            vmin=log_misfits.min(), vmax=log_misfits.max())
        scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=cm)

        mask = (
            (points[:,0]>=0.1)
            & (points[:,0]<=0.3)
            & (points[:,1]<=-0.1)
            & (points[:,1]>=-0.3))
        log_misfits_masked = log_misfits[mask]
        c_norm_masked = colors.Normalize(
            vmin=log_misfits_masked.min(), vmax=log_misfits_masked.max())
        scalar_map_masked = cmx.ScalarMappable(norm=c_norm_masked, cmap=cm)

        fig, axes = plt.subplots(1, 2, figsize=(9.5,4))

        voronoi_plot_2d(
            vor,
            show_vertices=False,
            line_colors='green',
            line_width=.5,
            point_size=2,
            ax=axes[0])
        voronoi_plot_2d(
            vor,
            show_vertices=False,
            line_colors='green',
            line_width=.5,
            point_size=2,
            ax=axes[1])

        # colorize
        for ireg, reg in enumerate(vor.regions):
            if not -1 in reg:
                ip_ = voronoi.find_point_of_region(vor, ireg)
                if ip_ == None:
                    continue
                point = vor.points[ip_]
                ips = np.where(
                    (points[:,0]==point[0])
                    & (points[:,1]==point[1]))
                if ips[0].shape[0] > 0:
                    ip = ips[0][0]
                    color = scalar_map.to_rgba(log_misfits[ip])
                    color_masked = scalar_map_masked.to_rgba(log_misfits[ip])

                    poly = [vor.vertices[i] for i in reg]
                    axes[0].fill(*zip(*poly), color=color)
                    axes[1].fill(*zip(*poly), color=color_masked)

        for ax in axes:
            ax.plot(0.2, -0.2, 'xr', markersize=6)
            ax.set_aspect('equal')
            ax.set(xlabel='dVs1 (km/s)', ylabel='dVs2 (km/s)')

        axes[0].set(xlim=[-0.5, 0.5], ylim=[-0.5,0.5])
        axes[1].set(xlim=[0.1, 0.3], ylim=[-0.3,-0.1])

        if 'title' in kwargs:
            fig.suptitle(kwargs['title'])
        plt.savefig(path, bbox_inches='tight')
        plt.close(fig)

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    n_cores = comm.Get_size()
    rank = comm.Get_rank()

    log = open('log_{}'.format(rank), 'w', buffering=1)

    na = NeighbouhoodAlgorithm.from_file(sys.argv[1], comm)

    if rank == 0:
        start_time = time.time_ns()

    log.write('Start running NA...\n')
    result = na.compute(comm, log)

    if rank == 0:
        end_time = time.time_ns()
        print(
            'NA finished in {} s'
            .format((end_time-start_time)*1e-9))

    log.close()

    if rank == 0:
        fig, ax = result.plot_models(
            types=[ParameterType.VSH], n_best=1,
            color='black', label='best model')
        _, ax = work_parameters.get_model_syntest1().plot(
            types=[ParameterType.VSH], ax=ax,
            color='red', label='input')
        _, ax = SeismicModel.ak135().plot(
            types=[ParameterType.VSH], ax=ax,
            color='gray', label='ak135')
        ax.set(
            ylim=[3479.5, 4000],
            xlim=[6.5, 8.])
        ax.legend()
        plt.savefig(
            'recovered_models_syntest1_nparam2_nspc256_80_test.pdf',
            bbox_inches='tight')
        plt.close(fig)
