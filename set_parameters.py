from parameters import Parameters


def set_parameters():

  return Parameters(
    dataset_trn_path    = '/media/guillaume/f0bb3659-b50a-4aac-b559-0953c567b645/afhq/train/wild',
    dataset_val_path    = '/media/guillaume/f0bb3659-b50a-4aac-b559-0953c567b645/afhq/val/wild',
    results_path        = '/media/guillaume/f0bb3659-b50a-4aac-b559-0953c567b645/RESULTS_DIFFUSION_MODEL/',
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
