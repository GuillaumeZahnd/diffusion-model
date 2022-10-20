import torch
import torch.nn.functional as F


def compute_loss(network_prediction, reference_target, loss_nickname):

  # Mean Square Error (L2)
  if loss_nickname == 'MSE':
    return F.mse_loss(network_prediction, reference_target)

  # Mean Absolute Error (L1)
  elif loss_nickname == 'MAE':
    return F.l1_loss(network_prediction, reference_target)

  # Huber
  elif loss_nickname == 'HUBER':
    return F.smooth_l1_loss(network_prediction, reference_target, beta=0.1)

  else:
    raise NotImplementedError()
