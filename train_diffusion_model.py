import torch
from unet import Unet
from icecream import ic

from set_parameters import set_parameters
from get_dataloader import get_dataloader


def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)




if __name__ == '__main__':

  p = set_parameters()

  # Model
  model = Unet(dim = p.IMAGE_SIZE)
  ic(count_parameters(model))


  # Device (GPU or CPU)
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model.to(device)


  # Optimizer
  if p.OPTIMIZER_NICKNAME == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr=p.LEARNING_RATE, momentum=p.MOMENTUM)
  elif p.OPTIMIZER_NICKNAME == 'ADAM':
    optimizer = torch.optim.Adam(model.parameters(), lr=p.LEARNING_RATE)

  # Scheduler
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, p.SCHEDULER_STEP_SIZE, p.SCHEDULER_GAMMA)

  model.train()

  # Dataloader
  loader_trn, nb_batches_trn, nb_samples_trn = get_dataloader(p)

  # Tractable diffusion

  """
  trn_loss_accumulator = EpochLossAccumulator()
  time_epoch = time.time()
  """

  for id_batch, (batch_names, batch_images) in enumerate(loader_trn):

    # Reset gradients to zero
    optimizer.zero_grad()

    # Pool a timestep value uniformally at random for every sample of the batch
    current_batch_size = batch_images.shape[0] # [N, C, H, W]
    timestep = torch.randint(0, p.NB_TIMESTEPS, (current_batch_size, ), device = device).long()

    # Noisify the image using the pooled timestep values
    noise = torch.randn_like(batch_images)

    batch_images_noisy = 

    # Forward pass with the deep neural network to predict the noise values

    # Calculate the corresponding loss, then back propagate the gradients to optimize the model weights


    """
    # Calculate loss, propagate gradients back through the model, and optimize
    loss = custom_loss(ima_net_batch, ima_ref_batch, p.LOSS_NICKNAME)
    loss.backward()
    optimizer.step()
    """

  """
    # Keep track of the calculated loss
    batch_size = sinogram_batch.shape[0]
    trn_loss_accumulator.update_losses(batch_size, loss.item()*batch_size)
    # Log the current batch
    log_batch_to_console_tensorboard_harddrive(
      id_epoch, 'Trn', trn_loss_accumulator, time_epoch, id_batch, nb_batch_trn, ima_ref_batch, ima_net_batch,
      file_name_batch, writer, flip_trn, p, print_epoch_dir)

  # Log the current epoch
  current_train_loss = trn_loss_accumulator.get_epoch_loss()
  log_epoch_console_tensorboard(
    writer, 'Trn', '0_training_loss', id_epoch, time_epoch, current_train_loss, new_val_loss)
  """
  # Learning rate evolution
  scheduler.step()

