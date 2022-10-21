import torch
import time

from epoch_loss_accumulator import EpochLossAccumulator
from print_batch_loss import print_batch_loss
from print_epoch_loss import print_epoch_loss
from forward_diffusion import q_sample
from compute_loss import compute_loss
from showcase_image import showcase_image


# TODO --> Log the loss on W&B instead of TensorBoard
def routine_trn(
  p,
  tdp,
  model,
  loader,
  id_epoch,
  optimizer,
  scheduler):

  # Training mode
  model.train()

  # Misc helpers
  trn_val_tst      = 'trn'
  time_epoch       = time.time()
  nb_batches       = len(loader)
  loss_accumulator = EpochLossAccumulator()

  for id_batch, (batch_names, batch_images) in enumerate(loader):

    # Determine the current batch size
    current_batch_size = batch_images.shape[0] # [N, C, H, W]

    # Reset gradients to zero
    optimizer.zero_grad()

    # Pool a timestep value uniformally at random for every sample of the batch
    batch_timesteps = torch.randint(0, p.NB_TIMESTEPS, (current_batch_size, ), device = p.DEVICE).long()

    # Noisify the image using the pooled timestep values
    batch_images_noisy, batch_target_noise = q_sample(
      batch_images.to(p.DEVICE), batch_timesteps, tdp.sqrt_alphas_cumprod, tdp.sqrt_one_minus_alphas_cumprod, noise=None)

    # Forward pass with the deep neural network to predict the noise values
    batch_predicted_noise = model(batch_images_noisy, batch_timesteps)

    # Loss
    loss = compute_loss(batch_predicted_noise, batch_target_noise, p.LOSS_NICKNAME)

    # Backpropagation
    loss.backward()

    # Optimize the model weights
    optimizer.step()

    # Keep track of the calculated loss
    loss_accumulator.update_losses(current_batch_size, loss.item())

    # Log the current batch
    showcase_image(batch_images, batch_images_noisy, batch_target_noise, batch_predicted_noise, id_epoch, id_batch, batch_names, 'trn', p.RESULTS_IMAGES_EPOCHS)
    print_batch_loss(id_epoch, trn_val_tst, loss_accumulator, time_epoch, id_batch, nb_batches)

  # Get epoch loss
  epoch_loss = loss_accumulator.get_epoch_loss()

  # Log the current epoch
  print_epoch_loss(
    trn_val_tst  = trn_val_tst,
    id_epoch     = id_epoch,
    epoch_loss   = epoch_loss,
    time_epoch   = time_epoch,
    new_val_loss = False)

  # Learning rate evolution
  scheduler.step()

  return model, optimizer, scheduler
