from pytomo.inversion.umc import UniformMonteCarlo
from pydsm.seismicmodel import SeismicModel
from pydsm.modelparameters import ModelParameters, ParameterType
import numpy as np
import matplotlib.pyplot as plt
from pydsm.event import Event
from pydsm.station import Station
from pydsm.utils.cmtcatalog import read_catalog
from pydsm.dataset import Dataset
from pydsm.dsm import PyDSMInput, compute, compute_models_parallel
from pydsm.windowmaker import WindowMaker
from mpi4py import MPI

def get_dataset(tlen=1634.8, nspc=64, sampling_hz=20):
    catalog = read_catalog()
    event = Event.event_from_catalog(
        catalog, '200707211534A')
    events = [event]
    stations = [
        Station('{:03d}'.format(i), 'DSM', i, 0.) for i in range(10,36)]
    dataset = Dataset.dataset_from_arrays(
        events, [stations], sampling_hz=sampling_hz)
    
    model = SeismicModel.prem()
    pydsm_input = PyDSMInput.input_from_arrays(
        event, stations, model, tlen, nspc, sampling_hz)
    pydsm_output = compute(pydsm_input, mode=2)
    pydsm_output.to_time_domain()
    dataset.data = pydsm_output.us[2] # select T component

    return dataset

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    n_cores = comm.Get_size()
    rank = comm.Get_rank()
    
    if rank == 0:
        depth_max = 900
        depth_min = 6371. - 6336.6 # Moho
        n_triangles = 18
        dr = np.round((depth_max - depth_min) / (n_triangles-1), 5)
        print("dr={}".format(dr))

        n_umc_samples = 4

        model_ref = SeismicModel.ak135()

        types = [ParameterType.VSH]
        radii = np.array([6371 - depth_max + i * dr for i in range(n_triangles)])
        model_params = ModelParameters(types, radii)
        range_dict = {ParameterType.VSH: [-0.2, 0.2]}

        umc = UniformMonteCarlo(
            model_ref, model_params, range_dict,
            mesh_type='triangle', seed=0)
        sample_models = umc.sample_models(n_umc_samples)
    else:
        sample_models = None

    tlen = 1634.8
    nspc = 128
    sampling_hz = 20

    dataset = get_dataset(tlen, nspc, sampling_hz)

    outputs = compute_models_parallel(
        dataset, sample_models, tlen, nspc, sampling_hz,
        comm)

    windows = WindowMaker.windows_from_dataset(
        dataset, 'prem', ['S'], t_before=10., t_after=40.)

    if rank == 0:
        misfit_dict = umc.process_outputs(
            outputs, dataset, sample_models, windows)
        print(misfit_dict)


        fig, ax = sample_models[0].plot(parameters=['vsh'])
        for sample in sample_models[1:]:
            sample.plot(ax=ax, parameters=['vsh'])
        model_ref.plot(ax=ax, parameters=['vsh'], color='black')
        ax.get_legend().remove()
        ax.set(ylim=[6371-depth_max, 6371], xlim=[3, 7])
        plt.show()