import torch
import math

from unet import Unet
from set_parameters              import set_parameters
from get_dataloaders             import get_dataloaders
from get_optimizer_and_scheduler import get_optimizer_and_scheduler
from routine_trn import routine_trn
from routine_val import routine_val
from package_utils.print_number_of_learnable_model_parameters import print_number_of_learnable_model_parameters
from tractable_diffusion_process import TractableDiffusionProcess


if __name__ == '__main__':

  # Parameters
  p = set_parameters()

  # Model
  model = Unet(dim = p.IMA_SIZE)
  print_number_of_learnable_model_parameters(model)

  # Device (GPU or CPU)
  model.to(p.DEVICE)

  # Optimizer and scheduler
  optimizer, scheduler = get_optimizer_and_scheduler(p, model)

  # Dataloader
  loader_trn, loader_val = get_dataloaders(p)

  # Tractable diffusion process
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
