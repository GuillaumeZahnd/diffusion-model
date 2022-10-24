import torch
import math

from get_optimizer_and_scheduler import get_optimizer_and_scheduler

from package_parameters.set_parameters             import set_parameters
from package_model.get_model                       import get_model
from package_dataloaders.get_dataloaders           import get_dataloaders
from package_diffusion.tractable_diffusion_process import TractableDiffusionProcess
from package_routines.routine_trn                  import routine_trn
from package_routines.routine_val                  import routine_val
from package_routines.routine_reverse_loop         import routine_reverse_loop
from package_routines.routine_reverse_generation   import routine_reverse_generation


if __name__ == '__main__':

  # Set parameters
  p = set_parameters()

  # Get model
  model = get_model(p)

  # Get optimizer and scheduler
  optimizer, scheduler = get_optimizer_and_scheduler(p, model)

  # Get dataloader
  loader_trn, loader_val = get_dataloaders(p)

  # Get and instance of the tractable diffusion process
  tdp = TractableDiffusionProcess(variance_schedule = p.VARIANCE_SCHEDULE, nb_timesteps = p.NB_TIMESTEPS)

  # Loop
  min_val_loss = math.inf
  for id_epoch in range(p.NB_EPOCHS):

    # Training routine
    model, optimizer, scheduler = routine_trn(
      p         = p,
      tdp       = tdp,
      model     = model,
      loader    = loader_trn,
      id_epoch  = id_epoch,
      optimizer = optimizer,
      scheduler = scheduler)

    # Validation routine
    min_val_loss = routine_val(
      p            = p,
      tdp          = tdp,
      model        = model,
      loader       = loader_val,
      id_epoch     = id_epoch,
      optimizer    = optimizer,
      scheduler    = scheduler,
      min_val_loss = min_val_loss)

    # Reverse diffusion
    if min_val_loss:
      routine_reverse_loop(
        p        = p,
        tdp      = tdp,
        model    = model,
        id_epoch = id_epoch)
      routine_reverse_generation(
        p        = p,
        tdp      = tdp,
        model    = model,
        id_epoch = id_epoch)
