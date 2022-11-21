import os
import sys
import torch

from package_model.get_model                       import get_model
from package_routines.routine_reverse_loop         import routine_reverse_loop
from package_routines.routine_reverse_generation   import routine_reverse_generation
from package_diffusion.tractable_diffusion_process import TractableDiffusionProcess


if __name__ == '__main__':

  # [Parameter] Root folder containing the results of previous experiments
  save_path = '/home/guillaume/RESULTS/diffusion-model'

  # [Parameter] Experiment name
  experiment_name = 'run_one'

  # Import the parameters
  sys.path.insert(0, os.path.join(save_path, experiment_name, 'backup_parameters'))
  from get_parameters import get_parameters

  # Get the experiment parameters
  p = get_parameters()

  # Get model
  model = get_model(p)

  # Get and instance of the tractable diffusion process
  tdp = TractableDiffusionProcess(p)

  # Load the trained model and set it in "eval" mode
  network_checkpoint_path_and_filename = os.path.join(
    save_path, experiment_name, 'trained_model', 'model_min_val_loss_{}.pt'.format(experiment_name))
  checkpoint = torch.load(network_checkpoint_path_and_filename, map_location = p.DEVICE)
  model.load_state_dict(checkpoint['model_state_dict'])
  model.eval()

  DO_REVERSE_LOOP = True
  if DO_REVERSE_LOOP:
    routine_reverse_loop(
      p     = p,
      tdp   = tdp,
      model = model,
      id_epoch = 999) # FIXME (display)

  DO_REVERSE_GENERATION = False
  if DO_REVERSE_GENERATION:
    routine_reverse_generation(
      p     = p,
      tdp   = tdp,
      model = model,
      id_epoch = 999) # FIXME (display)
