import math

from package_parameters.parameters import Parameters


def get_parameters():

  return Parameters(
    experiment_id       = 'run_one',
    dataset_trn_path    = '/home/guillaume/DATASETS/afhq/train/',
    dataset_val_path    = '/home/guillaume/DATASETS/afhq/val/',
    results_path        = '/home/guillaume/RESULTS/diffusion-model/',
    nb_samples_limit    = math.inf,
    rgb_or_grayscale    = 'rgb',
    ima_extension       = '.jpg',
    ima_size            = 128,
    batch_size          = 4,
    nb_epochs           = 100,
    optimizer_nickname  = 'SGD',
    learning_rate       = 1e-3,
    learning_momentum   = 0.99,
    scheduler_step_size = 1,
    scheduler_gamma     = 0.99,
    loss_nickname       = 'MAE',
    variance_schedule   = 'LINEAR',
    nb_timesteps        = 200)
