from package_model.unet import Unet
from package_loggers.print_number_of_learnable_model_parameters import print_number_of_learnable_model_parameters


def get_model(p):

  # Instanciate the model
  model = Unet(dim = p.IMA_SIZE, channels = 1 if p.RGB_OR_GRAYSCALE == 'grayscale' else 3)

  # Place the model to the adequate device (GPU or CPU)
  model.to(p.DEVICE)

  # Print the number of learnable parameters
  print_number_of_learnable_model_parameters(model)

  return model
