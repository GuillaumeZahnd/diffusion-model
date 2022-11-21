# README

## ...Work in progress...

![diffusion_model_1.png](../readme_images/diffusion_model_1.png?raw=true)

![diffusion_model_2.png](../readme_images/diffusion_model_2.png?raw=true)

![diffusion_model_3.png](../readme_images/diffusion_model_3.png?raw=true)

## About

- Methodology based on the original work from Ho et al. [1].
- Implementation based on the code provided by Ho et al. [2] and by Hugging Face [3].

> [1] Ho J., Jain A., and Abbeel P. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 33:6840-6851, 2020.
>
> [2] [https://github.com/hojonathanho/diffusion](https://github.com/hojonathanho/diffusion)
>
> [3] [https://huggingface.co/blog/annotated-diffusion](https://huggingface.co/blog/annotated-diffusion)


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

## Howto

### Run the training script

```sh
pipenv shell
python train_diffusion_model.py
```

### Run the inference script

```sh
pipenv shell
python demo_diffusion_model.py
```
