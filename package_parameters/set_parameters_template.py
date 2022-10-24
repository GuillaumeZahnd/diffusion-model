import math
from package_parameters.parameters import Parameters


def set_parameters():

  return Parameters(
    experiment_id       = '???',
    dataset_trn_path    = '???',
    dataset_val_path    = '???',
    results_path        = '???',
    nb_samples_limit    = math.inf,
    rgb_or_grayscale    = 'rgb',
    ima_extension       = '.jpg',
    ima_size            = 128,
    batch_size          = 4,
    nb_epochs           = 100,
    optimizer_nickname  = 'ADAM',
    learning_rate       = 1e-3,
    learning_momentum   = 0.99,
    scheduler_step_size = 1,
    scheduler_gamma     = 0.99,
    loss_nickname       = 'MSE',
    variance_schedule   = 'LINEAR',
    nb_timesteps        = 200)
