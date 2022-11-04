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

  # Resume training # TODO --> Add "USE_PRETRAINING" option as well
  if p.RESUME_TRAINING:
    checkpoint = torch.load(
      os.path.join(p.TRAINED_MODEL_PATH, 'model_min_val_loss_{}.pt'.format(p.EXPERIMENT_ID)), map_location = p.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    starting_epoch = checkpoint['epoch']

  else:
    starting_epoch = 0

  # TODO --> Properly handle "min_val_loss" if resume training
  min_val_loss = math.inf

  return model, optimizer, scheduler, starting_epoch, min_val_loss
