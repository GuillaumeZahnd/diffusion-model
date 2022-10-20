import torch
import torch.nn
import torch.nn.functional as F
from icecream import ic
from torchvision import transforms

from schedule import define_schedule
from dataloader import load_ima
from dataloader import trafo_pil_to_tensor
from dataloader import trafo_tensor_to_pil
from compute_loss import compute_loss



# ---
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

  return sqrt_alphas_cumprod_t * ima_input + sqrt_one_minus_alphas_cumprod_t * noise


# ----------------------------------------------------------------
if __name__ == '__main__':

  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  # Dataloader
  ima_zero = load_ima()
  ima = trafo_pil_to_tensor(ima_zero)
  ic(ima.shape)

  nb_timesteps = 200
  schedule_strategy = 'linear'

  # define beta schedule
  betas = define_schedule(schedule_strategy = schedule_strategy, nb_timesteps = nb_timesteps)

  # define alphas
  alphas = 1. - betas
  alphas_cumprod = torch.cumprod(alphas, axis=0)
  alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
  sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

  # calculations for diffusion q(x_t | x_{t-1}) and others
  sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
  sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

  # calculations for posterior q(x_{t-1} | x_t, x_0)
  posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

  t = torch.tensor([40])

  ima_noisy = q_sample(ima, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None)
  ima_noisy = trafo_tensor_to_pil(ima_noisy)

  import matplotlib.pyplot as plt
  plt.figure()
  plt.imshow(ima_noisy)
  plt.show()

