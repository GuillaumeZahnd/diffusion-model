import torch


# ----------------------------------------------------------------
def get_variance_schedule(variance_schedule, nb_timesteps):
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