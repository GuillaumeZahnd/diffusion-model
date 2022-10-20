from parameters import Parameters


def set_parameters():

  return Parameters(
    dataset_path        = '/media/guillaume/f0bb3659-b50a-4aac-b559-0953c567b645/afhq/train/wild',
    results_path        = '/media/guillaume/f0bb3659-b50a-4aac-b559-0953c567b645/RESULTS_DIFFUSION_MODEL/',
    image_extension     = '.jpg',
    image_size          = 128,
    batch_size          = 1,
    nb_epochs           = 5,
    optimizer_nickname  = 'ADAM',
    learning_rate       = 1e-3,
    learning_momentum   = 0.99,
    scheduler_step_size = 1,
    scheduler_gamma     = 0.99,
    loss_nickname       = 'MSE',
    variance_schedule   = 'LINEAR',
    nb_timesteps        = 200)
