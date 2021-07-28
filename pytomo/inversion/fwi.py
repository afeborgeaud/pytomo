from dsmpy.seismicmodel import SeismicModel
from dsmpy.modelparameters import ModelParameters, ParameterType
from dsmpy.windowmaker import WindowMaker
from dsmpy.window import Window
from dsmpy.component import Component
from dsmpy.utils.sklearnutils import get_XY, get_check_key
from dsmpy.dataset import Dataset
from dsmpy.dsm import compute_models_parallel
from pytomo.work.ca.params import get_dataset, get_model_syntest1_prem_vshvsv
from pytomo.inversion.cmcutils import process_outputs
from pytomo.utilities import get_temporary_str
from pytomo.inversion.inversionresult import FWIResult
from pytomo.utilities import minimum_tlen, minimum_nspc
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import logging


def freqency_hash(freq: float, freq2: float) -> str:
    """Return a hash to use as a key."""
    return f'{freq:.3f}_{freq2:.3f}'


def frequencies_from_hash(freq_hash: str) -> (float, float):
    """Return freq_min, freq_max from a frequency hash obtained
    using frequency_hash()."""
    hash_split = freq_hash.split('_')
    return float(hash_split[0]), float(hash_split[1])


class FWI:
    """Implement full-waveform inversion.

    Examples:
        >>> model_params = ModelParameters(
        ...    types=[ParameterType.VSH],
        ...    radii=[3480. + 20 * i for i in range(21)],
        ...    mesh_type='boxcar')
        >>> model_ref = SeismicModel.prem().boxcar_mesh(model_params)
        >>> dataset = Dataset.dataset_from_sac(
        ...    sac_files, headonly=False):
        >>> windows = WindowMaker.windows_from_dataset(
        ...    dataset, 'prem', ['ScS'], [Component.T],
        ...    t_before=20, t_after=40)
        >>> fwi = FWI(
        ...    model_ref, model_params, dataset,
        ...    windows, n_phases=1, mode=2)
        >>> model_1 = fwi.step(
        ...     model_ref, freq=0.01, freq2=0.04, n_pca_components=[4])
        >>> model_2 = fwi.step(
        ...     model_1, 0.01, 0.04, n_pca_components=[4])
        >>> model_3 = fwi.step(
        ...    model_2, 0.01, 0.08, n_pca_components=[4, 6, 8])
    """

    FILTER_TYPE = 'bandpass'
    logging.basicConfig(
        level=logging.INFO, filename='fwi.log', filemode='w')

    def __init__(
            self, model_ref, model_params, dataset_dict,
            windows_dict, n_phases, mode, phase_ref=None, buffer=10.):
        """

        Args:
            model_ref (SeismicModel): initial model
            model_params (ModelParams): model parameters
            dataset_dict (dict of Dataset): dict of filtered and cut
                datasets. The keys code the frequency ranges
                as given by frequency_hash()
            windows_dict (dict of list of Window): time windows
            n_phases (int): number of distinct phases in windows
            mode (int): computation mode.
                0: P-SV + SH, 1: P-SV, 2: SH
            phase_ref (str): reference phase used for static
                corrections
            buffer (float): buffer in seconds used to find
                the best shift for static correction
        """
        self.model_ref = model_ref
        self.model_params = model_params
        self.dataset_dict = dataset_dict
        self.windows_dict = windows_dict
        self.n_phases = n_phases
        self.mode = mode
        self.results = FWIResult(windows_dict)
        self.var = 2.5
        self.corr = 0.
        self.ratio = 2.5
        self.phase_ref = phase_ref
        self.buffer = buffer
        self.ignore_types = set()

    def set_selection(self, var: float, corr: float, ratio: float):
        """Set the variance, correlation, and ratio cutoffs
        used when deciding whether or not to keep a particular record."""
        self.var = var
        self.corr = corr
        self.ratio = ratio

    def set_ignore_types(self, parameter_types, ignore=True):
        """Do not update the parameter types.
        Switch these parameters on again by calling it
        with ignore=True"""
        if ignore:
            self.ignore_types = set(parameter_types)
        else:
            self.ignore_types -= set(parameter_types)


    def step(self, model, freq, freq2, n_pca_components=[4], alphas=[1.]):
        """Advance one step in the FWI iteration.

        Args:
            model (SeismicModel): the current initial model
            freq (float): minimum frequency for the bandpass filter
                (in Hz)
            freq2 (float): maximum frequency
            n_pca_components (list of int): number of PCA components
                to test (default is [4,])

        Returns:
            SeismicModel: the updated model

        """
        if MPI.COMM_WORLD.Get_rank() == 0:
            logging.info('Iteration step')
            logging.info(f'freq={freq}, freq2={freq2}')

        freq_hash = freqency_hash(freq, freq2)
        ds = self.dataset_dict[freq_hash]
        windows = self.windows_dict[freq_hash]
        window_npts = np.array(
            [window.get_length() * ds.sampling_hz
             for window in windows]
        ).astype('int').max()

        tlen = minimum_tlen(windows)
        nspc = minimum_nspc(tlen, freq2)

        if MPI.COMM_WORLD.Get_rank() == 0:
            logging.info(f'tlen={tlen}, nspc={nspc}')

        X, y, checked = get_XY(
            model, ds, windows, tlen=tlen, nspc=nspc,
            freq=freq, freq2=freq2, filter_type=FWI.FILTER_TYPE,
            sampling_hz=ds.sampling_hz, phase_ref=self.phase_ref,
            buffer=self.buffer, mode=self.mode)

        if MPI.COMM_WORLD.Get_rank() == 0:
            logging.info(
                f'Number of records used in inversion step: {len(checked)}')

            # switch parameters ON/OFF from self.ignore_types
            _, itypes, iradii = model.get_model_params().get_free_all_indices()
            parameter_types = [model.get_model_params().get_types()[i]
                               for i in itypes]
            parameter_radii = [model.get_model_params().get_grd_params()[i]
                               for i in iradii]
            mask = [False if p_type in self.ignore_types else True
                    for p_type in parameter_types]
            X = X[:, mask]
            used_types = [p_type
                          for p_type in model.get_model_params().get_types()
                          if p_type not in self.ignore_types]

            # TODO create a method for this piece of code
            # if QMU, sum all the layers to keep only a single QMU layer
            # this is because QMU cannot be well resolved
            if ParameterType.QMU in used_types:
                iqs = [i for i, p_type in enumerate(parameter_types)
                       if p_type == ParameterType.QMU]
                iq0 = iqs[0]
                i_keep = [i for i in range(X.shape[1])
                          if i not in iqs or i == iq0]
                i_iq0 = i_keep.index(iq0)
                X[:, iqs[0]] = X[:, iqs].sum(axis=1)
                X = X[:, i_keep]
                # place the single QMU parameter at the last column of X
                X[:, [i_iq0, -1]] = X[:, [-1, i_iq0]]
                used_types = [p_type for i, p_type in enumerate(used_types)
                              if i in i_keep]
                r_mean = np.mean([parameter_radii[i] for i in iqs])

            if ParameterType.QMU in used_types:
                # Do not include QMU in the PCA.
                # Instead, keep QMU features as is.
                pipe = Pipeline([
                    ('scaler', StandardScaler()),
                    ('reduce_dim', ColumnTransformer(
                        [
                            ('pca', PCA(), slice(-1)),
                            ('pass', 'passthrough', [-1])
                        ],
                    )
                    ),
                    ('reg', linear_model.Ridge())
                ])
            else:
                pipe = Pipeline([
                    ('scaler', StandardScaler()),
                    ('reduce_dim', PCA()),
                    ('reg', linear_model.Ridge())
                ])

            value_dicts = []
            rmses = []
            logging.info('Computing PCA')
            for n_pca in n_pca_components:
                for alpha in alphas:
                    if ParameterType.QMU in used_types:
                        pipe.set_params(
                            reduce_dim__pca__n_components=n_pca)
                    else:
                        pipe.set_params(
                            reduce_dim__n_components=n_pca)
                    pipe.fit(X, y)
                    y_pred = pipe.predict(X)
                    mse = mean_squared_error(y, y_pred)
                    rmse = np.sqrt(mse)
                    rmses.append(rmse)
                    logging.info(f'n_pca={n_pca}: rmse={rmse}')

                    coefs_trans = pipe['reg'].coef_.reshape(1, -1)

                    if ParameterType.QMU in used_types:
                        coefs_scaled = np.hstack((
                            pipe['reduce_dim'].transformers_[0][1]
                            .inverse_transform(
                                coefs_trans[:, :-1]),
                            coefs_trans[:, -1].reshape(1, -1))
                        )
                    else:
                        coefs_scaled = pipe['reduce_dim'].inverse_transform(
                            coefs_trans)

                    coefs = pipe['scaler'].transform(coefs_scaled)
                    best_params = np.array(coefs).flatten()

                    # TODO write a method for this piece of code
                    # rewrite the single QMU layers as n layers
                    if ParameterType.QMU in used_types:
                        n_grd_p = self.model_params.get_n_grd_params()
                        dq = best_params[-1]
                        Q = model.get_value_at(r_mean, ParameterType.QMU)
                        dQ = -Q**2 * dq
                        params_Q = np.repeat(dQ, n_grd_p - 1)
                        best_params = np.hstack((
                            best_params,
                            np.repeat(best_params[-1], n_grd_p - 1)
                        ))
                        # best_params[-n_grd_p:] /= n_grd_p

                    best_params = best_params.reshape(
                        len(used_types), -1)
                    best_params *= alpha
                    value_dicts.append(
                        {p_type: best_params[i]
                         for i, p_type in enumerate(
                            used_types)})
        else:
            value_dicts = None
        logging.info('Done')

        value_dicts = MPI.COMM_WORLD.bcast(value_dicts, root=0)
        updated_models = [
            model.multiply(self.model_params.get_values_matrix(value_dict))
            for value_dict in value_dicts
        ]
        outputs = compute_models_parallel(
            ds, updated_models, tlen=tlen,
            nspc=nspc, sampling_hz=ds.sampling_hz, mode=self.mode)
        if MPI.COMM_WORLD.Get_rank() == 0:
            selected_windows = [
                w for w in windows
                if get_check_key(w) in checked
            ]
            misfit_dict = process_outputs(
                outputs, ds, updated_models, selected_windows,
                freq, freq2, FWI.FILTER_TYPE
            )
        else:
            misfit_dict = None
        misfit_dict = MPI.COMM_WORLD.bcast(misfit_dict, root=0)
        variances = misfit_dict['variance'].mean(axis=1)
        logging.info(f'variances = {variances}')
        variances_per = (variances - variances.min()) / variances.min()

        # we select the model with the smallest number of PCA components
        # among the models with variance within 1% of the minimum
        # variance, to avoid overly complex models.
        k = np.argwhere(variances_per <= 0.01)[0][0]
        logging.info(f'Keeping model {k}')

        updated_model = updated_models[k]

        if MPI.COMM_WORLD.Get_rank() == 0:
            self.plot_model_log(
                updated_model, f'model{get_temporary_str()}.pdf')
            models_meta = [{'n_pca': n_pca_components[i], 'freq': freq,
                            'freq2': freq2, 'rmse': rmses[i],
                            'variance': variances[i]}
                           for i in range(len(n_pca_components))]
            self.results.add(
                updated_models, models_meta, misfit_dict, freq_hash)

        return updated_model

    def plot_model_log(self, model, figname: str):
        for p_type in self.model_params.get_types():
            fig, ax = plt.subplots(1, figsize=(4, 7))
            self.model_ref.plot(
                types=[p_type], ax=ax, label='ref')
            model.plot(
                types=[p_type], ax=ax, label='update')
            nodes = self.model_params.get_nodes()
            ax.set_ylim([nodes.min(), nodes.max()])
            if p_type in [ParameterType.VSH, ParameterType.VSV]:
                ax.set_xlim([6.5, 8])
            elif p_type == ParameterType.QMU:
                ax.set_xlim([100, 800])
            else:
                ax.set_xlim([6.5, 8])
            ax.legend()
            fig.savefig(str(p_type) + '_' + figname, bbox_inches='tight')
            plt.close(fig)


if __name__ == '__main__':
    types = [ParameterType.VSH, ParameterType.VSV]
    radii = [3480. + 20 * i for i in range(21)]
    model_params = ModelParameters(
        types=types,
        radii=radii,
        mesh_type='boxcar')
    model_ref = SeismicModel.prem().boxcar_mesh(model_params)

    t_before = 20.
    t_after = 40.
    sampling_hz = 20
    mode = 2
    freq = 0.01
    freq2 = 0.04
    phases = ['ScS']
    window_file = 'windows.pkl'

    dataset, output = get_dataset(
        get_model_syntest1_prem_vshvsv(), tlen=1638.4, nspc=256,
        sampling_hz=sampling_hz,
        mode=mode, add_noise=False, noise_normalized_std=1.)

    # windows = WindowMaker.windows_from_dataset(
    #     dataset, 'prem', phases, [Component.T, Component.R],
    #     t_before=t_before, t_after=t_after)

    windows = WindowMaker.load(window_file)
    windows_ScS = [w for w in windows if w.phase_name == 'ScS']
    windows_S = [w for w in windows if w.phase_name == 'S']
    windows_S_sS = [w for w in windows if w.phase_name in {'S', 'sS'}]
    windows_S_sS_trim = WindowMaker.set_limit(
        windows_ScS, t_before=5, t_after=15, inplace=False)
    windows_ScS_trimmed = WindowMaker.trim_windows(
        windows_ScS, windows_S_sS_trim)
    windows_ScS_proc = []
    for window in windows_ScS_trimmed:
        window_S = [
            w for w in windows_S
            if (w.station == window.station
                and w.event == window.event
                and w.component == window.component)
        ]
        if len(windows_S) == 1:
            windows_ScS_proc.append(
                Window(window.travel_time, window.event, window.station,
                       window.phase_name, window.component, window.t_before,
                       window.t_after, windows_S[0].t_shift)
            )

    fwi = FWI(
        model_ref, model_params, dataset, windows, 2, mode)

    model_1 = fwi.step(model_ref, 0.01, 0.04, n_pca_components=[8])
    model_2 = fwi.step(model_1, 0.01, 0.04, n_pca_components=[8])
    model_3 = fwi.step(model_2, 0.01, 0.08, n_pca_components=[8, 12, 16])
    model_4 = fwi.step(model_3, 0.01, 0.08, n_pca_components=[8, 12, 16])
    model_5 = fwi.step(model_4, 0.01, 0.08, n_pca_components=[12, 16, 20, 24])

    if MPI.COMM_WORLD.Get_rank() == 0:
        # save FWI result to object
        fname = 'fwiresult_' + get_temporary_str() + '.pkl'
        fwi.results.save(fname)

        # plot models
        fig, ax = plt.subplots(1)
        model_ref.plot(types=types, ax=ax, label='prem')
        get_model_syntest1_prem_vshvsv().plot(
            types=types, ax=ax, label='target')
        model_1.plot(types=types, ax=ax, label='it1')
        model_2.plot(types=types, ax=ax, label='it2')
        model_3.plot(types=types, ax=ax, label='it3')
        model_4.plot(types=types, ax=ax, label='it4')
        ax.set_ylim([3480, 4000])
        ax.set_xlim([6.5, 8])
        ax.legend()
        plt.show()
