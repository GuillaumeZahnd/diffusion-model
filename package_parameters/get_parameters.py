import math

from package_parameters.parameters import Parameters


def get_parameters():

  return Parameters(
    # Experiment
    experiment_id          = 'afhq_09', # Nickname to uniquely identify the run
    training_bootstrap     = 'NEW_TRAINING', # ['NEW_TRAINING', 'RESUME_TRAINING', 'USE_PRETRAINING']
    results_path           = 'home/guillaume/RESULTS/diffusion-model/', # Path to store the results
    pretraining_path       = '', # Path to the checkpoint of a pre-trained model
    pretraining_name       = '', # Name (including ".pt" extension) of the checkpoint of a pre-trained model
    # Dataset
    dataset_trn_path       = '/home/guillaume/DATASETS/stargan-v2/data/afhq_v2/train/', # Path to the training set folder
    dataset_val_path       = '/home/guillaume/DATASETS/stargan-v2/data/afhq_v2/test/', # Path to the validation set folder
    #dataset_trn_path       = '/home/guillaume/DATASETS/artbench/train/renaissance/', # Path to the training set folder
    #dataset_val_path       = '/home/guillaume/DATASETS/artbench/test/renaissance/', # Path to the validation set folder
    ima_extension          = '.png', # ['.jpg', '.png', ...] File format
    nb_samples_limit       = math.inf, # ['math.inf', 2000, ...] Maximal number of samples per epoch
    rgb_or_grayscale       = 'rgb', # ['RGB', 'GRAYSCALE'] Indicate whether images are to be treated as RGB or grayscale
    ima_size               = 128, # Image size after resizing
    cropping_method        = 'CENTER_CROP_THEN_RESIZE', # ['CENTER_CROP_THEN_RESIZE', 'RANDOM_CROP'] Indicate how images shall be cropped
    # Training
    backbone               = 'CONVNEXT', # ['RESNET', 'CONVNEXT']
    nb_epochs              = 500, # Number of training epochs
    batch_size             = 16, # Batch size
    optimizer_nickname     = 'SGD', # ['ADAM', 'SGD']
    learning_rate          = 0.005, # Learning rate
    learning_momentum      = 0.99, # Learning momentum
    scheduler_step_size    = 1, # Scheduler step size
    scheduler_gamma        = 0.99, # Scheduler gamma
    loss_nickname          = 'MAE', # ['MSE', 'MAE', 'HUBER'] Loss function
    # Diffusion
    nb_timesteps_learning  = 200, # Number of diffusion timesteps during the learning process, from 1 to T
    nb_timesteps_inference = 200, # Number of diffusion timesteps during the inference process, from 1 to T
    beta_one               = 1e-4, # Parametrization of the Gaussian distribution at initial timestep 1
    beta_t                 = 2e-2, # Parametrization of the Gaussian distribution at final timestep T
    variance_schedule      = 'LINEAR' # ['LINEAR', 'QUADRATIC', 'COSINE', 'SIGMOID'] Scheduler for the variance of the Gaussian distribution
    )
