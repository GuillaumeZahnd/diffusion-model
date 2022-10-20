import torch


# ----------------------------------------------------------------
def define_schedule(schedule_strategy, nb_timesteps):
  if schedule_strategy == 'COSINE':
    return cosine_beta_schedule(nb_timesteps)
  elif schedule_strategy == 'LINEAR':
    return linear_beta_schedule(nb_timesteps)
  elif schedule_strategy == 'QUADRATIC':
    return quadratic_beta_schedule(nb_timesteps)
  elif schedule_strategy == 'SIGMOID':
    return sigmoid_beta_schedule(nb_timesteps)
  else:
    raise NotImplementedError()


# ----------------------------------------------------------------
def cosine_beta_schedule(nb_timesteps, s=0.008):
  steps = nb_timesteps + 1
  x = torch.linspace(0, nb_timesteps, steps)
  alphas_cumprod = torch.cos(((x / nb_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
  alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
  betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
  return torch.clip(betas, 0.0001, 0.9999)


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
