import math

from package_parameters.parameters import Parameters


def get_parameters():

  return Parameters(
    experiment_id       = 'run_7',
    dataset_trn_path    = '/home/guillaume/DATASETS/afhq/train/',
    dataset_val_path    = '/home/guillaume/DATASETS/afhq/val/',
    results_path        = '/home/guillaume/TMP_RESULTS/DIFFUSION_MODEL/',
    nb_samples_limit    = math.inf,
    rgb_or_grayscale    = 'rgb',
    ima_extension       = '.jpg',
    ima_size            = 64,
    batch_size          = 32,
    nb_epochs           = 100,
    optimizer_nickname  = 'SGD',
    learning_rate       = 1e-3,
    learning_momentum   = 0.99,
    scheduler_step_size = 1,
    scheduler_gamma     = 0.99,
    loss_nickname       = 'MSE',
    variance_schedule   = 'COSINE',
    nb_timesteps        = 200)
