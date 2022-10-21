import torch
from package_utils.extract import extract


# Forward diffusion diffusion
def q_sample(ima_input, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None):

  # TODO --> When is "noise" not None?
  if noise is None:
    noise = torch.randn_like(ima_input)

  sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, ima_input.shape)
  sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, ima_input.shape)

  return sqrt_alphas_cumprod_t * ima_input + sqrt_one_minus_alphas_cumprod_t * noise, noise
