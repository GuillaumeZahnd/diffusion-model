import torch
import os
from pathlib import Path


class Parameters:

  def __init__(
    self,
    dataset_trn_path,
    dataset_val_path,
    results_path,
    ima_extension,
    ima_size,
    batch_size,
    nb_epochs,
    optimizer_nickname,
    learning_rate,
    learning_momentum,
    scheduler_step_size,
    scheduler_gamma,
    loss_nickname,
    variance_schedule,
    nb_timesteps):

    self.DATASET_TRN_PATH    = dataset_trn_path
    self.DATASET_VAL_PATH    = dataset_val_path
    self.RESULTS_PATH        = results_path
    self.IMA_EXTENSION       = ima_extension
    self.IMA_SIZE            = ima_size
    self.BATCH_SIZE          = batch_size
    self.NB_EPOCHS           = nb_epochs
    self.OPTIMIZER_NICKNAME  = optimizer_nickname
    self.LEARNING_RATE       = learning_rate
    self.LEARNING_MOMENTUM   = learning_momentum
    self.SCHEDULER_STEP_SIZE = scheduler_step_size
    self.SCHEDULER_GAMMA     = scheduler_gamma
    self.LOSS_NICKNAME       = loss_nickname
    self.VARIANCE_SCHEDULE   = variance_schedule
    self.NB_TIMESTEPS        = nb_timesteps

    # Set appropriate device (GPU if available, else CPU)
    self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create the directory structure if it does not already exists
    Path(self.RESULTS_PATH).mkdir(parents = True, exist_ok = True)
