from dsmpy import compute_dataset_parallel
from dsmpy.dataset import Dataset


def compute_dataset(dataset, model, tlen):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    compute_dataset_parallel(dataset, model,


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    tlen = 1638.4
    nspc = 256
    sampling_hz = 20
    mode = 2

    if rank == 0:
        sac_path = "/work/anselme/japan/DATA/2*/*T"
        sac_files = list(glob.iglob(sac_path))
        dataset = Dataset.dataset_from_sac(sac_files, headonly=False)

    outputs = compute_dataset_parallel(
        dataset, model, tlen, nspc, sampling_hz, comm, mode)



