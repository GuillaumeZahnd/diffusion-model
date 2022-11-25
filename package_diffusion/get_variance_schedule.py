import torch


# ----------------------------------------------------------------
def get_variance_schedule(p):
  if p.VARIANCE_SCHEDULE == 'COSINE':
    return cosine_beta_schedule(p.NB_TIMESTEPS, p.BETA_ONE, p.BETA_T)
  elif p.VARIANCE_SCHEDULE == 'LINEAR':
    return linear_beta_schedule(p.NB_TIMESTEPS, p.BETA_ONE, p.BETA_T)
  elif p.VARIANCE_SCHEDULE == 'QUADRATIC':
    return quadratic_beta_schedule(p.NB_TIMESTEPS, p.BETA_ONE, p.BETA_T)
  elif p.VARIANCE_SCHEDULE == 'SIGMOID':
    return sigmoid_beta_schedule(p.NB_TIMESTEPS, p.BETA_ONE, p.BETA_T)
  else:
    raise NotImplementedError()


# ----------------------------------------------------------------
# --> FIXME
def cosine_beta_schedule(nb_timesteps, beta_one, beta_t):
  steps = nb_timesteps + 1
  s = 0.008
  clip_low = 0.0001
  clip_high = 0.9999
  x = torch.linspace(0, nb_timesteps, steps)
  alphas_cumprod = torch.cos(((x / nb_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
  alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
  betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
  # return torch.clip(betas, clip_low, clip_high)
  return torch.clip(betas, clip_low, clip_high) * beta_t


# ----------------------------------------------------------------
def linear_beta_schedule(nb_timesteps, beta_one, beta_t):
  return torch.linspace(beta_one, beta_t, nb_timesteps)


# ----------------------------------------------------------------
def quadratic_beta_schedule(nb_timesteps, beta_one, beta_t):
  return torch.linspace(beta_one**0.5, beta_t**0.5, nb_timesteps) ** 2


# ----------------------------------------------------------------
def sigmoid_beta_schedule(nb_timesteps, beta_one, beta_t):
  betas = torch.linspace(-6, 6, nb_timesteps)
  return torch.sigmoid(betas) * (beta_t - beta_one) + beta_one
