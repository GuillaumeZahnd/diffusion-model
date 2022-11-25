import math

from package_parameters.parameters import Parameters


def get_parameters():

  return Parameters(
    experiment_id       = 'renaissance_02_sigmoid',
    dataset_trn_path    = '/home/guillaume/DATASETS/artbench/train/renaissance/',
    dataset_val_path    = '/home/guillaume/DATASETS/artbench/test/renaissance',
    results_path        = '/home/guillaume/RESULTS/diffusion-model/',
    nb_samples_limit    = math.inf,
    rgb_or_grayscale    = 'rgb',
    ima_extension       = '.jpg',
    ima_size            = 128,
    batch_size          = 4,
    nb_epochs           = 400,
    optimizer_nickname  = 'SGD',
    learning_rate       = 0.001,
    learning_momentum   = 0.99,
    scheduler_step_size = 1,
    scheduler_gamma     = 0.99,
    loss_nickname       = 'MSE',
    variance_schedule   = 'SIGMOID',
    beta_one            = 1e-4,
    beta_t              = 2e-2,
    nb_timesteps        = 200,
    resume_training     = False,
    use_pretraining     = False)
