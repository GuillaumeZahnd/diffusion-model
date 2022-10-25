import torch
from tqdm import tqdm

from package_utils.trafo_pil_to_and_from_tensor  import trafo_tensor_to_pil
from package_showcase.showcase_reverse_diffusion import showcase_reverse_diffusion


def routine_reverse_loop(p, tdp, model, id_epoch):

  nb_channels            = 1 if p.RGB_OR_GRAYSCALE == 'grayscale' else 3
  shape                  = (p.BATCH_SIZE, nb_channels, p.IMA_SIZE, p.IMA_SIZE)
  id_image_in_batch      = 0
  image_through_timesteps = []

  with torch.no_grad():

    img = torch.randn(shape, device = p.DEVICE)

    for id_timestep in tqdm(reversed(range(0, p.NB_TIMESTEPS)), desc = 'Reverse diffusion', total = p.NB_TIMESTEPS):
      img = tdp.p_sample(
        model = model,
        x = img,
        t = torch.full((p.BATCH_SIZE, ), id_timestep, device = p.DEVICE, dtype = torch.long),
        t_index = id_timestep)
      image_through_timesteps.append(trafo_tensor_to_pil(img, id_image_in_batch))

  showcase_reverse_diffusion(image_through_timesteps, p, id_epoch)
