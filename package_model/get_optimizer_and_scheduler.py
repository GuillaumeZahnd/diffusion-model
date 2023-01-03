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
  elif p.TRAINING_BOOTSTRAP == 'USE_PRETRAINING':

    # TODO --> Use the pretrained model of Experiment X to train the model of Experiment Y
    # TODO --> Ascertain that different learnable parameters can be used
    path_to_checkpoint = '/home/guillaume/RESULTS/diffusion-model/renaissance_04/trained_model/model_min_val_loss_renaissance_04.pt' # TODO
    checkpoint = torch.load(path_to_checkpoint, map_location = p.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'], strict = False)
    starting_epoch = 0
    min_val_loss = math.inf

    # TODO --> Example where the parameters related to the signal network are frozen
    """
    for name, param in network.named_parameters():
      if name[0:7] == 'net_sig':
        param.requires_grad = False
    """

  return model, optimizer, scheduler, starting_epoch, min_val_loss
