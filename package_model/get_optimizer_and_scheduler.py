import os
import math
import torch


def get_optimizer_and_scheduler(p, model):

  # Optimizer
  if p.OPTIMIZER_NICKNAME == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr = p.LEARNING_RATE, momentum = p.LEARNING_MOMENTUM)
  elif p.OPTIMIZER_NICKNAME == 'ADAM':
    optimizer = torch.optim.Adam(model.parameters(), lr = p.LEARNING_RATE)
  else:
    raise NotImplementedEror()

  # Scheduler
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, p.SCHEDULER_STEP_SIZE, p.SCHEDULER_GAMMA)

  # Start a new training from scratch
  if p.TRAINING_BOOTSTRAP == 'NEW_TRAINING':
    starting_epoch = 0
    min_val_loss = math.inf

  # Resume training (exact same network architecture, continues from the epoch with the min val loss)
  elif p.TRAINING_BOOTSTRAP == 'RESUME_TRAINING':
    checkpoint = torch.load(
      os.path.join(p.TRAINED_MODEL_PATH, 'model_min_val_loss_{}.pt'.format(p.EXPERIMENT_ID)), map_location = p.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    starting_epoch = checkpoint['epoch']
    min_val_loss = checkpoint['min_val_loss']

  # Use pretraining (Possibly different network architecture, possibly frozen parameters, starts at epoch zero
  # TODO --> Provide a snippet example with some frozen model parameters
  elif p.TRAINING_BOOTSTRAP == 'USE_PRETRAINING':
    checkpoint = torch.load(
      os.path.join(p.PRETRAINED_CHECKPOINT_PATH, p.PRETRAINED_CHECKPOINT_NAME),
      map_location = p.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'], strict = False)
    starting_epoch = 0
    min_val_loss = math.inf

  return model, optimizer, scheduler, starting_epoch, min_val_loss
