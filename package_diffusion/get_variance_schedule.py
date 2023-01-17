import torch


# ----------------------------------------------------------------
def get_variance_schedule(p):
  if p.VARIANCE_SCHEDULE == 'LINEAR':
    return linear_beta_schedule(p.NB_TIMESTEPS_LEARNING, p.BETA_ONE, p.BETA_T)
  elif p.VARIANCE_SCHEDULE == 'QUADRATIC':
    return quadratic_beta_schedule(p.NB_TIMESTEPS_LEARNING, p.BETA_ONE, p.BETA_T)
  elif p.VARIANCE_SCHEDULE == 'COSINE':
    return cosine_beta_schedule(p.NB_TIMESTEPS_LEARNING, p.BETA_ONE, p.BETA_T)
  elif p.VARIANCE_SCHEDULE == 'SIGMOID':
    return sigmoid_beta_schedule(p.NB_TIMESTEPS_LEARNING, p.BETA_ONE, p.BETA_T)
  else:
    raise NotImplementedError()


# ----------------------------------------------------------------
def linear_beta_schedule(nb_timesteps, beta_one, beta_t):
  return torch.linspace(beta_one, beta_t, nb_timesteps)


# ----------------------------------------------------------------
def quadratic_beta_schedule(nb_timesteps, beta_one, beta_t):
  return torch.linspace(beta_one**0.5, beta_t**0.5, nb_timesteps) ** 2


# ----------------------------------------------------------------
def cosine_beta_schedule(nb_timesteps, beta_one, beta_t):
  steps          = nb_timesteps + 1
  offset         = 1e-2         # Prevent betas from being too small near t=0
  clip_low       = 1e-3         # Prevent singularities at the start of the diffusion process near t=0
  clip_high      = 1 - clip_low # Prevent singularities at the end of the diffusion proces near t=T
  x              = torch.linspace(0, nb_timesteps, steps)
  alphas_cumprod = torch.cos(((x / nb_timesteps) + offset) / (1 + offset) * torch.pi * 0.5) ** 2
  alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
  betas          = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
  return torch.clip(betas, clip_low, clip_high) * beta_t


# ----------------------------------------------------------------
def sigmoid_beta_schedule(nb_timesteps, beta_one, beta_t):
  half_support = 7
  betas        = torch.linspace(-half_support, half_support, nb_timesteps)
  return torch.sigmoid(betas) * (beta_t - beta_one) + beta_one
