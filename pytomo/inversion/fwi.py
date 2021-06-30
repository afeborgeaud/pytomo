from dsmpy.seismicmodel import SeismicModel
from dsmpy.modelparameters import ModelParameters, ParameterType
from dsmpy.windowmaker import WindowMaker
from dsmpy.component import Component
from dsmpy.utils.sklearnutils import get_XY
from dsmpy.dataset import Dataset
from dsmpy.dsm import compute_models_parallel
from dsmpy import root_sac
from pytomo.work.ca.params import get_dataset, get_model_syntest1_prem_vshvsv
from pytomo.inversion.cmcutils import process_outputs
from pytomo.utilities import get_temporary_str
from pytomo.inversion.inversionresult import FWIResult
from pytomo.utilities import minimum_tlen, minimum_nspc
from pytomo.preproc.dataselection import compute_misfits
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from mpi4py import MPI
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import logging


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
            self, model_ref, model_params, dataset,
            windows, n_phases, mode):
        """

        Args:
            model_ref (SeismicModel): initial model
            model_params (ModelParams): model parameters
            dataset (Dataset): dataset
            windows (list of Window): time windows
            n_phases (int): number of distinct phases in windows
            mode (int): computation mode.
                0: P-SV + SH, 1: P-SV, 2: SH
        """
        self.model_ref = model_ref
        self.model_params = model_params
        self.dataset = dataset
        self.windows = windows
        self.n_phases = n_phases
        self.mode = mode
        self.results = FWIResult(windows)

    def step(self, model, freq, freq2, n_pca_components=[4]):
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

        ds = self.dataset.filter(
            freq, freq2, FWI.FILTER_TYPE, inplace=False)
        window_npts = int(np.array(
            [window.get_length() for window in self.windows]
        ).max() * ds.sampling_hz)
        ds.apply_windows(
            self.windows, self.n_phases, window_npts)

        tlen = minimum_tlen(self.windows)
        nspc = minimum_nspc(tlen, freq2)

        if MPI.COMM_WORLD.Get_rank() == 0:
            logging.info(f'tlen={tlen}, nspc={nspc}')

        X, y = get_XY(
            model, ds, self.windows, tlen=tlen, nspc=nspc,
            freq=freq, freq2=freq2, filter_type=FWI.FILTER_TYPE,
            sampling_hz=ds.sampling_hz, mode=self.mode)

        if MPI.COMM_WORLD.Get_rank() == 0:
            pipe = Pipeline(steps=[
                ('scaler', StandardScaler()),
                ('pca', PCA()),
                ('reg', linear_model.Ridge(alpha=0.))
            ])

            value_dicts = []
            rmses = []
            for n_pca in n_pca_components:
                pipe.set_params(pca__n_components=n_pca)
                pipe.fit(X, y)
                y_pred = pipe.predict(X)
                mse = mean_squared_error(y, y_pred)
                rmse = np.sqrt(mse)
                rmses.append(rmse)
                logging.info(f'n_pca={n_pca}, rmse={rmse}')

                coefs_trans = pipe['reg'].coef_.reshape(1, -1)
                coefs_scaled = pipe['pca'].inverse_transform(coefs_trans)
                coefs = pipe['scaler'].transform(coefs_scaled)

                best_params = np.array(coefs).reshape(len(types), -1)
                value_dicts.append(
                    {p_type: best_params[i]
                     for i, p_type in enumerate(types)})
        else:
            value_dicts = None

        value_dicts = MPI.COMM_WORLD.bcast(value_dicts, root=0)
        updated_models = [
            model.multiply(model_params.get_values_matrix(value_dict))
            for value_dict in value_dicts
        ]
        outputs = compute_models_parallel(
            ds, updated_models, tlen=tlen,
            nspc=nspc, sampling_hz=sampling_hz, mode=mode)
        if MPI.COMM_WORLD.Get_rank() == 0:
            for i in range(len(outputs)):
                for j in range(len(outputs[0])):
                    outputs[i][j].filter(freq, freq2, FWI.FILTER_TYPE)
            misfit_dict = process_outputs(
                outputs, ds, updated_models, windows)
        else:
            misfit_dict = None
        misfit_dict = MPI.COMM_WORLD.bcast(misfit_dict, root=0)
        variances = misfit_dict['variance'].mean(axis=1)
        logging.info(f'variances = {variances}')
        updated_model = updated_models[np.argmin(variances)]

        if MPI.COMM_WORLD.Get_rank() == 0:
            self.plot_model_log(
                updated_model, f'model{get_temporary_str()}.pdf')
            models_meta = [{'n_pca': n_pca_components[i], 'freq': freq,
                            'freq2': freq2, 'rmse': rmses[i],
                            'variance': variances[i]}
                           for i in range(len(n_pca_components))]
            self.results.add(updated_models, models_meta, misfit_dict)

        return updated_model

    def plot_model_log(self, model, figname: str):
        fig, ax = plt.subplots(1, figsize=(4, 7))
        self.model_ref.plot(
            types=self.model_params.get_types(), ax=ax, label='ref')
        get_model_syntest1_prem_vshvsv().plot(
            types=types, ax=ax, label='target')
        model.plot(
            types=self.model_params.get_types(), ax=ax, label='update')
        nodes = model_params.get_nodes()
        ax.set_ylim([nodes.min(), nodes.max()])
        ax.set_xlim([6.5, 8])
        ax.legend()
        fig.savefig(figname, bbox_inches='tight')


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

    dataset, output = get_dataset(
        get_model_syntest1_prem_vshvsv(), tlen=1638.4, nspc=256,
        sampling_hz=sampling_hz,
        mode=mode, add_noise=False, noise_normalized_std=1.)

    windows = WindowMaker.windows_from_dataset(
        dataset, 'prem', phases, [Component.T, Component.R],
        t_before=t_before, t_after=t_after)

    # misfits = compute_misfits(dataset, model_ref, windows, mode=mode)
    # if MPI.COMM_WORLD.Get_rank() == 0:
    #     fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    #     for i, (freq, freq2) in enumerate(misfits['frequency']):
    #         axes[i, 0].hist(misfits['misfit'][i]['variance'],
    #                         label='variance', bins=30)
    #         axes[i, 1].hist(misfits['misfit'][i]['corr'], label='corr',
    #                         bins=30)
    #         axes[i, 2].hist(misfits['misfit'][i]['ratio'], label='ratio',
    #                         bins=30)
    #     for ax in axes.ravel():
    #         ax.legend()
    #     plt.show()

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
