# pytomo
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)

Tools for seismic data processing and tomography in Python. Use [dsmpy](https://github.com/afeborgeaud/dsmpy) for exact 1-D waveform computation.<br>
Documentation and tutorials can be found [here](https://afeborgeaud.github.io/pytomo/).

# INSTALLATION

## Preferred method: dependencies using conda and building from source
1) Install [dsmpy](https://github.com/afeborgeaud/dsmpy). In the process, you should have created a ```dsm``` conda environment.
2) Clone the pytomo repository
```bash
git clone https://github.com/afeborgeaud/pytomo
```
3) Update the ```dsm``` conda environment (from step 1):
```bash
conda env update -n dsm --file environment.yml --prune
```
5) Install pytomo. ```/path/to/pytomo/``` is the path to the local pytomo git repository:
```bash
conda develop -n dsm /path/to/pytomo/
```

## Requirements
- Libraries for Python dev
```bash
sudo apt-get install python3-dev
```
- You might have to install GEOS lib dev, which are required by cartopy
```bash
sudo apt-get install libgeos-dev
```
```bash
sudo apt-get install libproj-dev proj-data proj-bin
```

## Using pip (building from source)
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
