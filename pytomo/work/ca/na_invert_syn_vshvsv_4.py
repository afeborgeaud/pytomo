import params as work_parameters
from pytomo.inversion.voronoi import plot_voronoi_2d
from pytomo.inversion.na import NeighbouhoodAlgorithm, InputFile
from dsmpy.modelparameters import ModelParameters, ParameterType
from dsmpy.seismicmodel import SeismicModel
from dsmpy.component import Component
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from mpi4py import MPI
import os
import sys

comm = MPI.COMM_WORLD
n_core = comm.Get_size()
rank = comm.Get_rank()

input_file = sys.argv[1]

# model parameters
n_nodes = 3
types = [ParameterType.VSH, ParameterType.VSV, ParameterType.RADIUS]
radii = np.linspace(3480., 3980., n_nodes, endpoint=True)
model_params = ModelParameters(types, radii, mesh_type='lininterp')
model = SeismicModel.prem().lininterp_mesh(model_params)

# set D'' layer to constant velocity and density
for r in radii[:-1]:
    izone = model.get_zone(r)
    for p_type in ParameterType.structure_types():
        v_dpp = model.get_value_at(radii[-1], p_type)
        model.set_value(
            izone, p_type, np.array([v_dpp, 0., 0., 0.]))

# constraints to parameters
mask_dict = dict()
for p_type in ParameterType.structure_types():
    mask_dict[p_type] = np.zeros(
        model_params.get_n_grd_params(), dtype='bool')
for p_type in types:
    mask_dict[p_type] = np.ones(
        model_params.get_n_grd_params(), dtype='bool')
# fix VSH and VSV for the shallowest node (grd_param index -2 and -1)
for p_type in [ParameterType.VSH, ParameterType.VSV]:
    mask_dict[p_type][[-2, -1]] = False
# fix the radius of the deepest node (grd_param index 0 and 1)
# (the CMB radius)
mask_dict[ParameterType.RADIUS][[0, 1]] = False

discon_arr = np.zeros(
    model_params.get_n_nodes(), dtype='bool')
# allows discontinuity at the middle node (node index 1)
# this represent the D" discontinuity
discon_arr[1] = True

model_params.set_constraints(
    mask_dict=mask_dict,
    discon_arr=discon_arr)

# parameter ranges
range_dict = dict()
for p_type in model_params.get_types():
    range_arr = np.empty((model_params.get_n_grd_params(), 2), dtype='float')
    if p_type != ParameterType.RADIUS:
        range_arr[:, 0] = -0.5
        range_arr[:, 1] = 0.5
    else:
        range_arr[:, 0] = -100
        range_arr[:, 1] = 100
    range_dict[p_type] = range_arr

# dataset
input = InputFile(input_file)
input_params = input.read()
dataset, _ = work_parameters.get_dataset_syntest_vshvsv_4(
    tlen=input_params['tlen'],
    nspc=input_params['nspc'],
    mode=input_params['mode'],
    add_noise=False,
    noise_normalized_std=2.)

na = NeighbouhoodAlgorithm.from_file(
    input_file, model, model_params, range_dict,
    dataset, comm)

# run
log_path = os.path.join(
    na.out_dir, 'log_{}'.format(rank))
log = open(log_path, 'w', buffering=1)

result = na.compute(comm, log)

# plot inverted model
if rank == 0:
    fig, ax = plt.subplots(1, 1, figsize=(4,6))
    result.plot_models(
        types=types, n_best=1, ax=ax,
        color='black', label='best model')
    work_parameters.get_model_syntest_vshvsv_4().plot(
        types=types, ax=ax,
        color='red', label='target')
    na.model_ref.plot(
        types=types, ax=ax,
        color='gray', label='ref',
        linestyle='dashed')
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

# compute best models
if rank == 0:
    indices_better = [0]
    for imod in range(1, result.meta['n_mod']):
        i_best = result.get_indices_best_models(n_best=1, n_mod=imod+1)
        if i_best != indices_better[-1]:
            print(i_best, imod)
            indices_better.append(imod)
    print(indices_better)
    print(len(result.models))
    models = [result.models[i] for i in indices_better]
else:
    models = None
outputs = result.compute_models(models)

# plot results
if rank == 0:
    out_dir = result.meta['out_dir']
    points = NeighbouhoodAlgorithm._get_points_for_voronoi(
                result.perturbations, range_dict, model_params._types)
    points = points[:, model_params.get_free_indices()]
    # points = np.array(result.perturbations)[
    # :, model_params.get_free_indices()]
    misfits = result.misfit_dict['variance'].mean(axis=1)
    n_r = result.meta['n_r']
    n_s = result.meta['n_s']
    n_mod = points.shape[0]
    i_out = 0
    for imod in range(n_mod):
        figpath = os.path.join(
            out_dir, 'voronoi_{:05d}.png'.format(imod))
        fig = plt.figure(figsize=(17,5))
        gs = gridspec.GridSpec(1, 4, width_ratios=[2, 1, 1, 1])
        # plot voronoi cells
        ax0 = fig.add_subplot(gs[0])
        _, _, colormap = plot_voronoi_2d(
            points[:imod+1, [1, 4]], misfits[:imod+1],
            xlim=[-0.5, 0.5], ylim=[-0.5, 0.5], ax=ax0)
        ax0.set(
            xlabel='dVSH (km/s)',
            ylabel='dVSV (km/s)')
        fig.colorbar(
            colormap, ax=ax0, label='Variance',
            shrink=0.5, fraction=0.07, pad=0.15,
            orientation='horizontal')
        # plot model
        ax1 = fig.add_subplot(gs[1])
        na.model_ref.plot(
            types=types, ax=ax1,
            color='gray', label='ref',
            linestyle='dashed')
        result.plot_models(
            types=[ParameterType.VSH], n_best=1, n_mod=imod+1, ax=ax1,
            color='black', label='best model')
        result.plot_models(
            types=[ParameterType.VSV], n_best=1, n_mod=imod+1, ax=ax1,
            color='black', label='best model', linestyle='dashdot')
        work_parameters.get_model_syntest_vshvsv_4().plot(
            types=[ParameterType.VSH], ax=ax1,
            color='cyan', label='target')
        work_parameters.get_model_syntest_vshvsv_4().plot(
            types=[ParameterType.VSV], ax=ax1,
            color='red', label='target')
        ax1.set(
            ylim=[3480, 4000],
            xlim=[6.5, 8.])
        ax1.legend(loc='upper right')
        pos1 = list(ax1.get_position().bounds)
        pos1[0] -= 0.03
        ax1.set_position(pos1)
        # plot waveforms
        # R
        ax2 = fig.add_subplot(gs[2])
        if (i_out+1 < len(indices_better)
            and indices_better[i_out+1] == imod):
            i_out += 1
        result.plot_event(
            outputs[i_out], 0, ax2, component=Component.R,
            color='red')
        pos2 = list(ax2.get_position().bounds)
        pos2[0] -= 0.03
        ax2.set_position(pos2)
        ax2.set_title('Radial')
        # T
        ax3 = fig.add_subplot(gs[3])
        if (i_out+1 < len(indices_better)
            and indices_better[i_out+1] == imod):
            i_out += 1
        result.plot_event(
            outputs[i_out], 0, ax3, component=Component.T)
        pos3 = list(ax3.get_position().bounds)
        pos3[0] -= 0.03
        ax3.set_position(pos3)
        ax3.set_title('Transverse')

        fig.suptitle('Model #{}'.format(imod))
        plt.savefig(
            figpath, bbox_inches='tight', dpi=250)
        plt.close(fig)

log.close()
