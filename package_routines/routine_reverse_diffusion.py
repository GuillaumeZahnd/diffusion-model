import torch
from tqdm import tqdm

from package_utils.trafo_pil_to_and_from_tensor  import trafo_tensor_to_pil
from package_utils.get_subsampled_interval       import get_subsampled_interval
from package_showcase.showcase_reverse_diffusion import showcase_reverse_diffusion


def routine_reverse_diffusion(p, tdp, model, id_epoch = None):

  nb_channels       = 1 if p.RGB_OR_GRAYSCALE == 'grayscale' else 3
  shape             = (p.BATCH_SIZE, nb_channels, p.IMA_SIZE, p.IMA_SIZE)
  id_image_in_batch = 0
  idx_sub2ini       = get_subsampled_interval(p.NB_TIMESTEPS_LEARNING, p.NB_TIMESTEPS_INFERENCE)

  # Store "p.NB_TIMESTEPS+1" images: The first (t=T) is the pure noise image, the last (t=0) is the generated image
  img_through_timesteps = []

  with torch.no_grad():

    img = torch.randn(shape, device = p.DEVICE)
    img_through_timesteps.append(trafo_tensor_to_pil(img, id_image_in_batch))

    for id_timestep_sub in tqdm(
      reversed(range(0, p.NB_TIMESTEPS_INFERENCE)),
      desc = 'Reverse diffusion',
      total = p.NB_TIMESTEPS_INFERENCE):

      id_timestep_ini = idx_sub2ini[id_timestep_sub]

      img = tdp.p_sample(
        model   = model,
        x       = img,
        t       = torch.full((p.BATCH_SIZE, ), id_timestep_ini, device = p.DEVICE, dtype = torch.long),
        t_index = id_timestep_ini)

      img_through_timesteps.append(trafo_tensor_to_pil(img, id_image_in_batch))

  # Re-order the list so the first item corresponds to t=0 (generated image) and the last item to t=T (pure noise)
  img_through_timesteps.reverse()

  showcase_reverse_diffusion(
    img_through_timesteps = img_through_timesteps,
    p                     = p,
    id_epoch              = id_epoch)
