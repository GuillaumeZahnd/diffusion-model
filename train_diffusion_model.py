from package_parameters.set_parameters             import set_parameters
from package_model.get_model                       import get_model
from package_model.get_optimizer_and_scheduler     import get_optimizer_and_scheduler
from package_dataloaders.get_dataloaders           import get_dataloaders
from package_diffusion.tractable_diffusion_process import TractableDiffusionProcess
from package_utils.initiate_wandb                  import initiate_wandb
from package_routines.routine_trn                  import routine_trn
from package_routines.routine_val                  import routine_val
from package_routines.routine_reverse_diffusion    import routine_reverse_diffusion
from package_routines.routine_reverse_generation   import routine_reverse_generation


if __name__ == '__main__':

  # Set parameters
  p = set_parameters()

  # Get model
  model = get_model(p)

  # Get optimizer and scheduler
  model, optimizer, scheduler, starting_epoch, min_val_loss = get_optimizer_and_scheduler(p, model)

  # Get dataloader
  loader_trn, loader_val = get_dataloaders(p)

  # Get and instance of the tractable diffusion process
  tdp = TractableDiffusionProcess(p)

  # Instantiate a run on Weights and Biases
  initiate_wandb()

  # Loop
  for id_epoch in range(starting_epoch, p.NB_EPOCHS):

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
    min_val_loss, new_val_loss = routine_val(
      p            = p,
      tdp          = tdp,
      model        = model,
      loader       = loader_val,
      id_epoch     = id_epoch,
      optimizer    = optimizer,
      scheduler    = scheduler,
      min_val_loss = min_val_loss)

    if new_val_loss:
      # Reverse diffusion
      routine_reverse_diffusion(
        p        = p,
        tdp      = tdp,
        model    = model,
        id_epoch = id_epoch)

      # Reverse generation
      routine_reverse_generation(
        p        = p,
        tdp      = tdp,
        model    = model,
        id_epoch = id_epoch)
