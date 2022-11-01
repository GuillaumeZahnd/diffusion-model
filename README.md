# README

## Prerequisite

- CUDA-capable GPU
- NVIDIA drivers and CUDA toolkit already installed
- Python 3.x

### Install pip

```
python -m pip install --upgrade pip
```

### Install pipenv

```sh
pip install --user pipenv
```

### Install the dependencies in the virtual environment

```sh
python setup_venv.py
```

### Install torch and torchvision via pip rather than via the Pipfile

```sh
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

### Create and edit a Git-untracked file to describe the experiment parameters

```sh
cp package_parameters/set_parameters_tempate.py package_parameters/set_parameters.py
vim package_parameters/set_parameters.py
```

### How to run the script

```sh
pipenv shell
python train_diffusion_model.py
```
