import numpy as np
import glob
from mpi4py import MPI
from dsmpy.dataset import Dataset
from dsmpy.seismicmodel import SeismicModel
from dsmpy.component import Component
from dsmpy.windowmaker import WindowMaker

if __name__ == '__main__':
    sac_files = glob.glob(
        '/work/anselme/CA_ANEL_NEW/VERTICAL/200707211534A/*T')
    dataset = Dataset.dataset_from_sac(sac_files, headonly=False)
    model = SeismicModel.prem()
    tlen = 3276.8
    nspc = 1024
    sampling_hz = 20
    freq = 0.005
    freq2 = 0.167

    comm = MPI.COMM_WORLD
    n_cores = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        windows_S = WindowMaker.windows_from_dataset(
            dataset, 'prem', ['s', 'S', 'Sdiff'],
            [Component.T], t_before=10., t_after=20.)
        windows_P = WindowMaker.windows_from_dataset(
            dataset, 'prem', ['p', 'P', 'Pdiff'],
            [Component.Z], t_before=10., t_after=20.)
        windows = windows_S #+ windows_P

        dataset = Dataset.dataset_from_sac(sac_files, headonly=False)
    else:
        dataset = None

    
