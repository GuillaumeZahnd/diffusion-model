import torch
import os
from pathlib import Path
from package_parameters.backup_parameters            import backup_parameters
from package_loggers.print_all_experiment_parameters import print_all_experiment_parameters


class Parameters:

  # ----------------------------------------------------------------
  def __init__(
    self,
    experiment_id,
    training_bootstrap,
    results_path,
    dataset_trn_path,
    dataset_val_path,
    ima_extension,
    nb_samples_limit,
    rgb_or_grayscale,
    ima_size,
    cropping_method,
    nb_epochs,
    batch_size,
    optimizer_nickname,
    learning_rate,
    learning_momentum,
    scheduler_step_size,
    scheduler_gamma,
    loss_nickname,
    nb_timesteps,
    beta_one,
    beta_t,
    variance_schedule,
    ):
    
    # Experiment
    self.EXPERIMENT_ID       = experiment_id
    self.TRAINING_BOOTSTRAP  = training_bootstrap
    self.RESULTS_PATH        = results_path
    
    # Dataset
    self.DATASET_TRN_PATH    = dataset_trn_path
    self.DATASET_VAL_PATH    = dataset_val_path
    self.IMA_EXTENSION       = ima_extension
    self.NB_SAMPLES_LIMIT    = nb_samples_limit
    self.RGB_OR_GRAYSCALE    = rgb_or_grayscale
    self.IMA_SIZE            = ima_size
    self.CROPPING_METHOD     = cropping_method
    
    # Training
    self.NB_EPOCHS           = nb_epochs
    self.BATCH_SIZE          = batch_size
    self.OPTIMIZER_NICKNAME  = optimizer_nickname
    self.LEARNING_RATE       = learning_rate
    self.LEARNING_MOMENTUM   = learning_momentum
    self.SCHEDULER_STEP_SIZE = scheduler_step_size
    self.SCHEDULER_GAMMA     = scheduler_gamma
    self.LOSS_NICKNAME       = loss_nickname
    
    # Diffusion
    self.VARIANCE_SCHEDULE   = variance_schedule
    self.BETA_ONE            = beta_one
    self.BETA_T              = beta_t
    self.NB_TIMESTEPS        = nb_timesteps
    
    # Set appropriate device (GPU if available, else CPU)
    self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set the filename of the trained model
    self.TRAINED_MODEL_NAME = 'model_min_val_loss_' + self.EXPERIMENT_ID + '.pt'

    # Define and create directory structure
    self.define_and_create_directory_structure()

    # Backup the experiment parameters
    backup_parameters(self.BACKUP_PARAMETERS_PATH)

    # Print all experiment parameters
    print_all_experiment_parameters(self)


  # ----------------------------------------------------------------
  def define_and_create_directory_structure(self):
    self.RESULTS_IMAGES_EPOCHS = os.path.join( # TODO --> Use better names
      self.RESULTS_PATH, self.EXPERIMENT_ID, 'trn_val_tst_images_across_epochs')
    self.TRAINED_MODEL_PATH = os.path.join(
      self.RESULTS_PATH, self.EXPERIMENT_ID, 'trained_model')
    self.BACKUP_PARAMETERS_PATH = os.path.join(
      self.RESULTS_PATH, self.EXPERIMENT_ID, 'backup_parameters')

    Path(self.RESULTS_IMAGES_EPOCHS).mkdir(parents = True, exist_ok = True)
    Path(self.TRAINED_MODEL_PATH).mkdir(parents = True, exist_ok = True)
    Path(self.BACKUP_PARAMETERS_PATH).mkdir(parents = True, exist_ok = True)
