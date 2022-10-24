import torch
from tqdm import tqdm
from package_utils.extract import extract
from package_utils.trafo_pil_to_and_from_tensor import trafo_tensor_to_pil


# ----------------------------------------------------------------
def p_sample(model, x, t, t_index, tdp):

  betas_t = extract(tdp.betas, t, x.shape)
  sqrt_one_minus_alphas_cumprod_t = extract(tdp.sqrt_one_minus_alphas_cumprod, t, x.shape)
  sqrt_recip_alphas_t = extract(tdp.sqrt_recip_alphas, t, x.shape)

  # Equation 11 in the paper
  # Use our model (noise predictor) to predict the mean
  model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)

  if t_index == 0:
    return model_mean
  else:
    posterior_variance_t = extract(tdp.posterior_variance, t, x.shape)
    noise = torch.randn_like(x)
    # Algorithm 2 line 4:
    return model_mean + torch.sqrt(posterior_variance_t) * noise


# Algorithm 2 (including returning all images)
# ----------------------------------------------------------------
def p_sample_loop(model, shape, tdp, nb_timesteps):
  device = next(model.parameters()).device

  b = shape[0]
  # start from pure noise (for each example in the batch)
  img = torch.randn(shape, device=device)
  imgs = []

  for id_timestep in tqdm(reversed(range(0, nb_timesteps)), desc='sampling loop time step', total=nb_timesteps):
    img = p_sample(model, img, torch.full((b,), id_timestep, device=device, dtype=torch.long), id_timestep, tdp)
    imgs.append(trafo_tensor_to_pil(img[0,:,:,:]))
  return imgs


# ----------------------------------------------------------------
def sample(model, image_size, batch_size, channels, nb_timesteps, tdp, id_epoch, results_path):
  with torch.no_grad():
    images_over_timesteps = p_sample_loop(model, shape=(batch_size, channels, image_size, image_size), tdp=tdp, nb_timesteps=nb_timesteps)


    import matplotlib.pyplot as plt
    import math
    import os

    fig, ax = plt.subplots(10, 20)

    fig.set_dpi(300)
    fig.set_size_inches(20*3, 10*3, forward = True)

    [axx.set_axis_off() for axx in ax.ravel()]

    for idx in range(nb_timesteps):
      id_x = idx % 20
      id_y = math.floor(idx / 20)
      ima = images_over_timesteps[idx]

      axx = ax[id_y, id_x]
      fig.sca(axx)
      axx.set_title(idx)
      im = axx.imshow(ima)

    fig.savefig(os.path.join(results_path, str(id_epoch) + '.png'), bbox_inches = 'tight')
    plt.close()

