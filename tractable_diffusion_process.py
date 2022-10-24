import torch
import torch.nn.functional as F
from package_utils.extract import extract


class TractableDiffusionProcess:

  def __init__(self, variance_schedule, nb_timesteps):

    # Define betas schedule
    self.betas = define_schedule(variance_schedule = variance_schedule, nb_timesteps = nb_timesteps)

    # Derive alphas
    self.alphas              = 1. - self.betas
    self.alphas_cumprod      = torch.cumprod(self.alphas, axis=0)
    self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
    self.sqrt_recip_alphas   = torch.sqrt(1.0 / self.alphas)

    # calculations for diffusion q(x_t | x_{t-1}) and others
    self.sqrt_alphas_cumprod           = torch.sqrt(self.alphas_cumprod)
    self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)


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


# ----------------------------------------------------------------
def define_schedule(variance_schedule, nb_timesteps):
  if variance_schedule == 'COSINE':
    return cosine_beta_schedule(nb_timesteps)
  elif variance_schedule == 'LINEAR':
    return linear_beta_schedule(nb_timesteps)
  elif variance_schedule == 'QUADRATIC':
    return quadratic_beta_schedule(nb_timesteps)
  elif variance_schedule == 'SIGMOID':
    return sigmoid_beta_schedule(nb_timesteps)
  else:
    raise NotImplementedError()


# ----------------------------------------------------------------
def cosine_beta_schedule(nb_timesteps):
  steps = nb_timesteps + 1
  s = 0.008
  clip_low = 0.0001
  clip_high = 0.9999
  x = torch.linspace(0, nb_timesteps, steps)
  alphas_cumprod = torch.cos(((x / nb_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
  alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
  betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
  return torch.clip(betas, clip_low, clip_high)


# ----------------------------------------------------------------
def linear_beta_schedule(nb_timesteps):
  beta_start = 0.0001
  beta_end = 0.02
  return torch.linspace(beta_start, beta_end, nb_timesteps)


# ----------------------------------------------------------------
def quadratic_beta_schedule(nb_timesteps):
  beta_start = 0.0001
  beta_end = 0.02
  return torch.linspace(beta_start**0.5, beta_end**0.5, nb_timesteps) ** 2


# ----------------------------------------------------------------
def sigmoid_beta_schedule(nb_timesteps):
  beta_start = 0.0001
  beta_end = 0.02
  betas = torch.linspace(-6, 6, nb_timesteps)
  return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
