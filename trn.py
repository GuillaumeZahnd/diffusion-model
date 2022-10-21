import torch
from forward_diffusion import q_sample
from compute_loss import compute_loss


import time
from epoch_loss_accumulator import EpochLossAccumulator
from print_batch_loss import print_batch_loss
from print_epoch_loss import print_epoch_loss

def trn(
  p,
  model,
  loader,
  optimizer,
  scheduler,
  sqrt_alphas_cumprod,
  sqrt_one_minus_alphas_cumprod,
  id_epoch,
  nb_batch_trn):

  model.train()

  trn_loss_accumulator = EpochLossAccumulator()
  time_epoch = time.time()
  trn_val_tst = 'trn'

  for id_batch, (batch_names, batch_images) in enumerate(loader):

    # Determine the current batch size
    current_batch_size = batch_images.shape[0] # [N, C, H, W]

    # Reset gradients to zero
    optimizer.zero_grad()

    # Pool a timestep value uniformally at random for every sample of the batch
    batch_timesteps = torch.randint(0, p.NB_TIMESTEPS, (current_batch_size, ), device = p.DEVICE).long()

    # Noisify the image using the pooled timestep values
    batch_images_noisy, noise = q_sample(
      batch_images.to(p.DEVICE), batch_timesteps, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None)

    # Forward pass with the deep neural network to predict the noise values
    predicted_noise = model(batch_images_noisy, batch_timesteps)

    # Loss
    loss = compute_loss(predicted_noise, noise, p.LOSS_NICKNAME)

    # Backpropagation
    loss.backward()

    # Optimize the model weights
    optimizer.step()

    # Keep track of the calculated loss
    trn_loss_accumulator.update_losses(current_batch_size, loss.item()*current_batch_size)

    # Log the current batch
    print_batch_loss(id_epoch, trn_val_tst, trn_loss_accumulator, time_epoch, id_batch, nb_batch_trn)
    """
    log_batch_to_console_tensorboard_harddrive(
      id_epoch, 'Trn', trn_loss_accumulator, time_epoch, id_batch, nb_batch_trn, ima_ref_batch, ima_net_batch,
      file_name_batch, writer, flip_trn, p, print_epoch_dir)
    """

  # Log the current epoch
  print_epoch_loss(trn_val_tst, id_epoch, loss, time_epoch, new_val_loss=False)
  """
  current_train_loss = trn_loss_accumulator.get_epoch_loss()
  log_epoch_console_tensorboard(
    writer, 'Trn', '0_training_loss', id_epoch, time_epoch, current_train_loss, new_val_loss)
  """

  # Learning rate evolution
  scheduler.step()

  return model, optimizer, scheduler
