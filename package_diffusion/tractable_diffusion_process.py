import torch
import torch.nn.functional as F

from package_diffusion.get_variance_schedule import get_variance_schedule
from package_utils.extract                   import extract


class TractableDiffusionProcess:

  def __init__(self, p):

    # Define betas schedule
    self.betas = get_variance_schedule(p)

    # Derive alphas
    self.alphas              = 1 - self.betas
    self.alphas_cumprod      = torch.cumprod(self.alphas, axis = 0)
    self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value = 1)
    self.sqrt_recip_alphas   = torch.sqrt(1 / self.alphas)

    # calculations for diffusion q(x_t | x_{t-1}) and others
    self.sqrt_alphas_cumprod           = torch.sqrt(self.alphas_cumprod)
    self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)


  # ----------------------------------------------------------------
  # Forward diffusion process
  def q_sample(self, ima_input, t, noise = None):

    # TODO --> When is "noise" not None?
    if noise is None:
      noise = torch.randn_like(ima_input)

    sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, ima_input.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, ima_input.shape)

    return sqrt_alphas_cumprod_t * ima_input + sqrt_one_minus_alphas_cumprod_t * noise, noise


  # ----------------------------------------------------------------
  # Reverse diffusion process
  def p_sample(self, model, x, t, t_index):

    betas_t                         = extract(self.betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t             = extract(self.sqrt_recip_alphas, t, x.shape)

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)

    if t_index == 0:
      return model_mean
    else:
      posterior_variance_t = extract(self.posterior_variance, t, x.shape)
      noise = torch.randn_like(x)
      # Algorithm 2 line 4:
      return model_mean + torch.sqrt(posterior_variance_t) * noise
