from pytomo.inversion.inversionresult import InversionResult
from pytomo.inversion import modelutils
from pytomo.inversion.umcutils import UniformMonteCarlo
import pytomo.inversion.voronoi as voronoi
from pytomo import utilities
from pydsm.seismicmodel import SeismicModel
from pydsm.modelparameters import ModelParameters, ParameterType
from pydsm.event import Event
from pydsm.station import Station
from pydsm.utils.cmtcatalog import read_catalog
from pydsm.dataset import Dataset
from pydsm.dsm import PyDSMInput, compute, compute_models_parallel
from pydsm.windowmaker import WindowMaker
from pydsm.component import Component
import numpy as np
from mpi4py import MPI
from scipy.spatial import voronoi_plot_2d, Voronoi
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import time
import sys
import os
import glob
import warnings
import copy

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
            values = line.strip().split()[1:]
            ss = [s.strip() for s in values]
            value_parsed = ss
        elif key == 'components':
            values = line.strip().split()[1:]
            ss = [s.strip() for s in values]
            value_parsed = [Component.parse_component(s) for s in ss]
        elif key == 'seed':
            value_parsed = int(value)
        elif key == 'convergence_threshold':
            value_parsed = float(value)
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
            self, dataset, model_ref, model_params, range_dict, tlen, nspc,
            sampling_hz, mode, n_mod, n_s, n_r,
            phases, components, t_before, t_after, filter_type,
            freq, freq2, sac_files, distance_min, distance_max,
            convergence_threshold,
            stf_catalog, result_path, seed, verbose, comm):
        self.dataset = dataset
        self.model_ref = model_ref
        self.model_params = model_params
        self.range_dict = range_dict
        self.tlen = tlen
        self.nspc = nspc
        self.sampling_hz = sampling_hz
        self.mode = mode
        self.n_mod = n_mod
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
        self.convergence_threshold = convergence_threshold
        self.stf_catalog = stf_catalog
        self.seed = seed
        self.verbose = verbose
        self.comm = comm

        self.rng_gibbs = np.random.default_rng(seed)

        if comm.Get_rank() == 0:
            out_dir = 'output_' + utilities.get_temporary_str()
            os.mkdir(out_dir)
        else:
            out_dir = None

        self.out_dir = comm.bcast(out_dir, root=0)
        self.result_path = os.path.join(self.out_dir, result_path)

    @classmethod
    def from_file(
        cls, input_file_path, model_ref, model_params,
        range_dict, dataset, comm):
        '''Build NeighbourhoodAlgorithm object from input file and
        key inputs
        Args:
        Returns:
            na (NeighbourhoodAlgorithm):
        '''
        params = InputFile(input_file_path).read()

        tlen = params['tlen']
        nspc = params['nspc']
        sampling_hz = params['sampling_hz']
        mode = params['mode']
        n_mod = params['n_mod']
        n_s = params['n_s']
        n_r = params['n_r']
        phases = params['phases']
        components = params['components']
        t_before = params['t_before']
        t_after = params['t_after']
        filter_type = params['filter_type']
        freq = params['freq']
        freq2 = params['freq2']
        sac_files = None
        distance_min = params['distance_min']
        distance_max = params['distance_max']
        stf_catalog = params['stf_catalog']
        result_path = params['result_path']
        seed = params['seed']
        verbose = params['verbose']
        convergence_threshold = params['convergence_threshold']

        # fix 
        dataset.tlen = tlen
        dataset.nspc = nspc

        return cls(
            dataset, model_ref, model_params, range_dict, tlen, nspc,
            sampling_hz, mode, n_mod, n_s, n_r,
            phases, components, t_before, t_after, filter_type,
            freq, freq2, sac_files, distance_min, distance_max,
            convergence_threshold,
            stf_catalog, result_path, seed, verbose, comm)

    def get_meta(self):
        '''Return meta parameters for the NA inversion'''
        return dict(
            range_dict=self.range_dict, tlen=self.tlen, nspc=self.nspc,
            sampling_hz=self.sampling_hz, mode=self.mode, n_mod=self.n_mod,
            n_s=self.n_s, n_r=self.n_r, phases=self.phases,
            components=self.components, t_before=self.t_before,
            t_after=self.t_after, filter_type=self.filter_type,
            freq=self.freq, freq2=self.freq2, sac_files=self.sac_files,
            distance_min=self.distance_min, distance_max=self.distance_max,
            convergence_threshold=self.convergence_threshold,
            stf_catalog=self.stf_catalog, result_path=self.result_path,
            seed=self.seed, verbose=self.verbose, out_dir=self.out_dir)

    @classmethod
    def from_file_with_default(cls, input_file_path, comm):
        '''Build object from the input file and default values
        Args:
            input_file_path (str): path of the input file
            comm (mpi4py.Comm_world): MPI communicator
        Returns:
            na (NeighbourhoodAlgorithm):
        '''
        params = InputFile(input_file_path).read()

        # define default model parameters
        types = [ParameterType.VSH]
        n_upper_mantle = 0
        n_mtz = 0
        n_lower_mantle = 0
        n_dpp = 4
        model_ref, model_params = modelutils.std_boxcar_mesh(
            n_upper_mantle, n_mtz, n_lower_mantle, n_dpp, types,
            verbose=params['verbose'])

        # define default parameter ranges
        range_dict = dict()
        for param_type in types:
            range_arr = np.empty(
                (model_params._n_grd_params, 2), dtype='float')
            range_arr[:, 0] = -0.5
            range_arr[:, 1] = 0.5
            range_dict[param_type] = range_arr

        # create dataset
        dataset = Dataset.dataset_from_sac(
            params['sac_files'], headonly=False)

        return cls.from_file(
            input_file_path, model_ref, model_params,
            range_dict, dataset, comm)

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

    @staticmethod
    def get_points_for_voronoi(perturbations, range_dict, types):
        '''
        Args:
            perturbations (list(ndarray)): list of model perturbations
        Return:
            points: points in a n_grd_params*len(types) dimensional
                space to define the voronoi cells
        '''
        scale_arr = np.hstack(
            [range_dict[p][:,1] - range_dict[p][:,0]
            for p in types])

        points = np.array(perturbations)
        points /= scale_arr

        return points

    def get_bounds_for_voronoi(self):
        min_bounds = np.zeros(
            self.model_params._n_grd_params*len(self.model_params._types),
            dtype='float')
        max_bounds = np.zeros(
            self.model_params._n_grd_params*len(self.model_params._types),
            dtype='float')
        for itype in range(len(self.model_params._types)):
            for igrd in range(self.model_params._n_grd_params):
                i = igrd + itype * self.model_params._n_grd_params
                min_bounds[i] = self.range_dict[
                    self.model_params._types[itype]][igrd, 0]
                max_bounds[i] = self.range_dict[
                    self.model_params._types[itype]][igrd, 1]
                # scale
                min_bounds[i] /= (max_bounds[i] - min_bounds[i])
                max_bounds[i] /= (max_bounds[i] - min_bounds[i])
        return min_bounds, max_bounds

    def compute_one_step(
            self, umcutils, dataset, models, perturbations,
            result, windows, comm):
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
            result.add_result(models, misfit_dict, perturbations)

    def compute(self, comm, log=None):
        '''Run the inversion.
        Args:
            comm (mpi4py.Comm_world): MPI Communicator
        Returns:
            result (InversionResult): inversion results
        '''
        rank = comm.Get_rank()
        if log is not None:
            log.write('Start running NA...\n')

        n_distinct_comp_phase = len(self.phases)

        free_indices = self.model_params.get_free_indices()
        print('free_indices: {}'.format(free_indices))

        n_pass = self.n_mod // self.n_s
        assert self.n_mod % self.n_s == 0
        if self.verbose > 1:
            print('n_pass={}'.format(n_pass))

        if rank == 0:
            scale_arr = np.hstack(
                [self.range_dict[p][:,1] - self.range_dict[p][:,0] 
                for p in self.model_params._types])

            umcutils = UniformMonteCarlo(
                self.model_ref, self.model_params, self.range_dict,
                mesh_type='lininterp', seed=self.seed)
            models, perturbations = umcutils.sample_models(self.n_s)
        else:
            models = None
            perturbations = None
            umcutils = None

        self.range_dict = comm.bcast(self.range_dict, root=0)
        
        if rank == 0:
            if self.filter_type is not None:
                self.dataset.filter(self.freq, self.freq2, self.filter_type)

            windows = self.get_windows(self.dataset)

            npts_max = int((self.t_before+self.t_after) * self.sampling_hz)
            self.dataset.apply_windows(
                windows, n_distinct_comp_phase, npts_max, buffer=0.)
        else:
            windows = None
            self.dataset = None
        self.dataset = comm.bcast(self.dataset, root=0)

        result = InversionResult(
                    self.dataset, windows, self.get_meta())

        if rank == 0:
            start_time = time.time_ns()

        # step 0
        self.compute_one_step(
            umcutils, self.dataset, models, perturbations, result,
            windows, comm)
        comm.Barrier()

        # steps 1,...,n_pass-1
        ipass = 1
        converged = False
        while (ipass < n_pass) and not converged:
            print(rank, ipass)
            if rank == 0:
                # indexing of points corrsespond to that of 
                # perturbations and of that of models
                points = NeighbouhoodAlgorithm.get_points_for_voronoi(
                    result.perturbations, self.range_dict,
                    self.model_params._types)

                min_bounds, max_bounds = self.get_bounds_for_voronoi()
                indices_best = umcutils.get_best_models(
                    result.misfit_dict, self.n_r)

                models = []
                perturbations = []
                for imod in range(self.n_r):
                    ip = indices_best[imod]
                    # current_model = result.models[ip]
                    current_perturbations = np.array(result.perturbations[ip])

                    value_dict = dict()
                    for i, param_type in enumerate(self.model_params._types):
                        value_dict[param_type] = current_perturbations[
                            (i*self.model_params._n_grd_params)
                            :((i+1)*self.model_params._n_grd_params)]

                    n_step = int(self.n_s//self.n_r)

                    current_point = np.array(points[ip])
                    self.model_params.it = 0 # just to be sure
                    for istep in range(n_step):
                        idim, itype, igrd = self.model_params.get_it_indices()
                        self.model_params.it += 1
                        # print(istep, idim, itype, igrd)

                        # calculate bound
                        points_free = points[:, free_indices]
                        current_point_free = current_point[free_indices]
                        idim_free = np.where(free_indices==idim)[0][0]

                        tmp_bounds1 = voronoi.implicit_find_bound_for_dim(
                            points_free, points_free[ip],
                            current_point_free, idim_free, n_nearest=300,
                            min_bound=min_bounds[idim],
                            max_bound=max_bounds[idim], step_size=0.001,
                            n_step_max=1000, log=log)
                        tmp_bounds2 = voronoi.implicit_find_bound_for_dim(
                            points_free, points_free[ip],
                            current_point_free, idim_free, n_nearest=500,
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

                        lo, up = bounds

                        # per = self.rng_gibbs.uniform(lo, up, 1)[0]
                        per = self.bi_triangle(lo, up)
                        
                        scale = (
                            self.range_dict[
                                self.model_params._types[itype]][igrd, 1]
                            - self.range_dict[
                                self.model_params._types[itype]][igrd, 0])
                        value_dict[
                            self.model_params._types[itype]][igrd] += per*scale
                        
                        # account for constraints
                        for param_type in self.model_params._types:
                            for jgrd in range(self.model_params._n_grd_params):
                                index = (
                                    self.model_params.equal_dict[
                                        param_type][jgrd])
                                if index != jgrd:
                                    value_dict[param_type][jgrd] = (
                                        value_dict[param_type][index])

                        per_arr = np.hstack(
                            [value_dict[p] for p in self.model_params._types])

                        # TODO implements
                        value_dict_m = copy.deepcopy(value_dict)
                        # value_dict_m[ParameterType.VSH][1] = 0.

                        new_model = self.model_ref.build_model(
                            umcutils.mesh, self.model_params,
                            value_dict, value_dict_m)
                        models.append(new_model)

                        if log:
                            log.write(
                                '{} {} {} {} {} {} {:.3f} {:.3f} {:.3f}\n'
                                .format(rank, ipass, imod, istep, idim,
                                        per_arr, lo, up, per))

                        # print(per_arr)
                        # current_point += per_arr / scale_arr
                        # current_point[idim] += per
                        current_point = per_arr / scale_arr

                        perturbations.append(per_arr)

                        istep += 1
            else:
                models = None

            self.compute_one_step(
                umcutils, self.dataset, models, perturbations,
                result, windows, comm)

            # check convergence
            if rank == 0:
                if ipass > 2:
                    perturbations_diff = result.get_model_perturbations_diff(
                        self.n_r, scale=scale_arr, smooth=True, n_s=self.n_s)
                    perturbations_diff_free = perturbations_diff[
                        :, free_indices]

                    converged = (
                        (perturbations_diff_free[-2:] 
                        <= self.convergence_threshold).all())

            converged = comm.bcast(converged, root=0)

            ipass += 1
            comm.Barrier()

        if rank == 0:
            end_time = time.time_ns()
            result.save(self.result_path)
            if self.verbose > 0:
                if log is not None:
                    log.write('Models and misfits computation done in {} s\n'
                        .format((end_time-start_time) * 1e-9))
                    log.write('Results saved to \'{}\''.format(self.result_path))
                else:
                    print('Models and misfits computation done in {} s'
                        .format((end_time-start_time) * 1e-9))
                    print('Results saved to \'{}\''.format(self.result_path))

            conv_curve_path = os.path.join(
                self.out_dir, 'convergence_curve.pdf')
            self.save_convergence_curve(
                conv_curve_path, result, scale_arr, free_indices, smooth=True)

            var_curve_path = os.path.join(
                self.out_dir, 'variance_curve.pdf')
            self.save_variance_curve(var_curve_path, result, smooth=True)

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
        if a==b==0:
            return 0.
        assert (a < b) and (a <= 0) and (b >= 0)
        x_unif = self.rng_gibbs.uniform(0, 1, 1)[0]
        x = self.bi_triangle_cfd_inv(x_unif, a, b)
        return x

    @staticmethod
    def plot_voronoi_2d(
            points, misfits,
            xlim=None, ylim=None,
            ax=None, **kwargs):
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
        x_max = points[:, 0].max()
        x_min = points[:, 0].min()
        y_max = points[:, 1].max()
        y_min = points[:, 1].min()
        points_ = np.append(
            points,
            # [[x_min*10, y_min*10], [x_min*10, y_max*10],
            #  [x_max*10, y_min*10], [x_max*10, y_max*10]],
            [[-9999, -9999], [-9999, 9999], [9999, -9999], [9999, 9999]],
            axis = 0)
        vor = Voronoi(points_)

        # color map
        # log_misfits = np.log(misfits)
        log_misfits = np.array(misfits)
        cm = plt.get_cmap('hot')
        c_norm  = colors.Normalize(
            vmin=log_misfits.min(), vmax=log_misfits.max())
        # c_norm = colors.Normalize(vmin=0., vmax=0.3)
        scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=cm)

        if ax is None:
            fig, ax = plt.subplots(1, 1)
        else:
            fig = None

        voronoi_plot_2d(
            vor,
            show_vertices=False,
            line_colors='green',
            line_width=.5,
            point_size=2,
            ax=ax)

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

                    poly = [vor.vertices[i] for i in reg]
                    ax.fill(*zip(*poly), color=color)

        ax.plot(0.2, 0., '*c', markersize=8)
        ax.set_aspect('equal')

        if xlim is None:
            xlim = [x_min, x_max]
            ylim = [y_min, y_max]
        ax.set(xlim=xlim, ylim=ylim)

        if ('title' in kwargs) and (fig is not None):
            fig.suptitle(kwargs['title'])
        
        return fig, ax, scalar_map

    def save_convergence_curve(
            self, path, result, scale_arr, free_indices, smooth=True):
        perturbations_diff = result.get_model_perturbations_diff(
                        self.n_r, scale=scale_arr, smooth=True, n_s=self.n_s)
        perturbations_diff_free = perturbations_diff[
            :, free_indices].max(axis=1)

        x = np.arange(1, perturbations_diff_free.shape[0]+1)
        
        fig, ax = plt.subplots(1)
        ax.plot(x, perturbations_diff_free, '-xk')
        if smooth:
            ax.set(
                xlabel='Iteration #',
                ylabel='Mean model perturbation (km/s)')
        else:
            ax.set(
                xlabel='Model #',
                ylabel='Model perturbation (km/s)')
        plt.savefig(path, bbox_inches='tight')
        plt.close(fig)
    
    def save_variance_curve(self, path, result, smooth=True):
        variances = result.get_variances(smooth=smooth, n_s=self.n_s)
        x = np.arange(1, variances.shape[0]+1)

        fig, ax = plt.subplots(1)
        ax.plot(x, variances, '-xk')
        if smooth:
            ax.set(
                xlabel='Iteration #',
                ylabel='Mean variance')
        else:
            ax.set(
                xlabel='Model #',
                ylabel='Variance')
        plt.savefig(path, bbox_inches='tight')
        plt.close(fig)

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    n_cores = comm.Get_size()
    rank = comm.Get_rank()

    na = NeighbouhoodAlgorithm.from_file_with_default(sys.argv[1], comm)

    log_path = os.path.join(
        na.out_dir, 'log_{}'.format(rank))
    log = open(log_path, 'w', buffering=1)

    if rank == 0:
        start_time = time.time_ns()

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
        _, ax = na.model_ref.plot(
            types=[ParameterType.VSH], ax=ax,
            color='gray', label='ref')
        ax.set(
            ylim=[3480, 4000],
            xlim=[6.5, 8.])
        ax.legend()
        fig_path = os.path.join(
            na.out_dir, 'inverted_models.pdf')
        plt.savefig(
            fig_path,
            bbox_inches='tight')
        plt.close(fig)
