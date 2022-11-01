# README

## Prerequisite

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
