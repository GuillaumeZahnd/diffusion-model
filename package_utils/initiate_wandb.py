import wandb
import os


# TODO --> Parametrize more finely
def initiate_wandb():
  wandb.init(project = 'diffusion-model', entity = 'gzahnd')
  wandb.save(os.path.join('package_parameters', 'get_parameters.py'))
