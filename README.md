# pytomo

Tools for seismic data processing and tomography in Python. Use [pydsm](https://github.com/afeborgeaud/pydsm) for exact 1-D waveform computation based on the Fortran Direct Solution Method (DSM).

# Installation

You'll need the following dependencies, which can be installed using the Conda package manager:
```shell
# create environment pytomo
conda create -n pydsm
# install dependencies
conda install -n pydsm numpy mpi4py pandas matplotlib -y
conda install -n pydsm -c conda-forge obspy geographiclib -y
# activate env
conda activate pydsm
```
