def print_number_of_learnable_model_parameters(model):
  print('Number of learnable model parameters: {:,}'.format(
    sum(param_idx.numel() for param_idx in model.parameters() if param_idx.requires_grad)))
