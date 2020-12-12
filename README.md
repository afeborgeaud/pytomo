# pytomo
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)

Tools for seismic data processing and tomography in Python. Use [dsmpy](https://github.com/afeborgeaud/dsmpy) for exact 1-D waveform computation.

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
pip install dist/*.whl
```
or
```
pip install dist/*.tar.gz
```
