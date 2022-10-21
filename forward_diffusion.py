import torch


# ----------------------------------------------------------------
def extract(a, t, x_shape):
  batch_size = t.shape[0]
  out = a.gather(-1, t.cpu())
  return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


# ----------------------------------------------------------------
# Forward diffusion
def q_sample(ima_input, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None):

  # TODO --> When is "noise" not None?
  if noise is None:
    noise = torch.randn_like(ima_input)

  sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, ima_input.shape)
  sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, ima_input.shape)

  return sqrt_alphas_cumprod_t * ima_input + sqrt_one_minus_alphas_cumprod_t * noise, noise
