import nn
import torch.nn.functional as F


class ForwardDiffusion(nn.Module):

  def __init__(self, nb_timesteps):
    super().__init__()

    self.nb_timesteps = timesteps

    beta_start = 0.01 / self.nb_timesteps
    beta_end   = 0.1 / self.nb_timesteps
    betas      = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

    alphas              = 1 - betas
    alphas_cumprod      = torch.cumprod(alphas, axis=0)
    #alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
