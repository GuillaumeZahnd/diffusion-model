import math

from package_parameters.parameters import Parameters


def set_parameters():

  return Parameters(
    experiment_id       = '???',                     #
    dataset_trn_path    = '???',                     #
    dataset_val_path    = '???',                     #
    results_path        = '???',                     #
    nb_samples_limit    = math.inf,                  #
    rgb_or_grayscale    = 'rgb',                     #
    ima_extension       = '.jpg',                    #
    ima_size            = 128,                       #
    cropping_method     = 'CENTER_CROP_THEN_RESIZE', # ['CENTER_CROP_THEN_RESIZE', 'RANDOM_CROP']
    batch_size          = 4,                         #
    nb_epochs           = 100,                       #
    optimizer_nickname  = 'ADAM',                    # ['ADAM', 'SGD']
    learning_rate       = 1e-3,                      #
    learning_momentum   = 0.99,                      #
    scheduler_step_size = 1,                         #
    scheduler_gamma     = 0.99,                      #
    loss_nickname       = 'MSE',                     #
    variance_schedule   = 'LINEAR',                  #
    beta_one            = 1e-4,                      #
    beta_t              = 2e-2,                      #
    nb_timesteps        = 1000,                      #
    training_bootstrap  = 'NEW_TRAINING')            # ['NEW_TRAINING', 'RESUME_TRAINING', 'USE_PRETRAINING']
