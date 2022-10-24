import torch
import os
from pathlib import Path


class Parameters:

  def __init__(
    self,
    experiment_id,
    dataset_trn_path,
    dataset_val_path,
    results_path,
    nb_samples_limit,
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

    self.EXPERIMENT_ID       = experiment_id
    self.DATASET_TRN_PATH    = dataset_trn_path
    self.DATASET_VAL_PATH    = dataset_val_path
    self.RESULTS_PATH        = results_path
    self.NB_SAMPLES_LIMIT    = nb_samples_limit
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
    self.RESULTS_IMAGES_EPOCHS = os.path.join(
      self.RESULTS_PATH, self.EXPERIMENT_ID, 'trn_val_tst_images_across_epochs')
    self.RESULTS_TRAINED_MODEL = os.path.join(
      self.RESULTS_PATH, self.EXPERIMENT_ID, 'trained_model')

    Path(self.RESULTS_IMAGES_EPOCHS).mkdir(parents = True, exist_ok = True)
    Path(self.RESULTS_TRAINED_MODEL).mkdir(parents = True, exist_ok = True)
