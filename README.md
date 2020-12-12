# pytomo

Tools for seismic data processing and tomography in Python. Use [pydsm](https://github.com/afeborgeaud/pydsm) for exact 1-D waveform computation based on the Fortran Direct Solution Method (DSM).

# Installation

### Build from source
1) Clone the pytomo repository
```
git clone https://github.com/afeborgeaud/pytomo
```
2) (Optional) You may want to install pytomo in a virtual environment. If so, do
```
python3 -m venv venv
source activate ./venv/bin/activate
```
3) Install [*build*](https://pypi.org/project/build/), a PEP517 package builder
```
pip install build
```
4) To build the pytomo package, from the root directory ```pytomo``` run
```
python -m build .
```
5) This creates ```.whl``` and ```.gz.tar``` dist files in the ```dist``` directory. Now pydsm can be installed with
```
pip install dist/pytomo-1.0a0-py3-none-any.whl
```
or
```
pip install dist/pytomo-1.0a0.tar.gz
```
