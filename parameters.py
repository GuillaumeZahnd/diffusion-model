import torch


class Parameters:

  def __init__(
    self,
    dataset_path,
    results_path,
    image_extension,
    image_size,
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

    self.DATASET_PATH        = dataset_path
    self.RESULTS_PATH        = results_path
    self.IMAGE_EXTENSION     = image_extension
    self.IMAGE_SIZE          = image_size
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

    self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
