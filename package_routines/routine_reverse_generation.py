import torch
from tqdm import tqdm

from package_utils.trafo_pil_to_and_from_tensor   import trafo_tensor_to_pil
from package_showcase.showcase_reverse_generation import showcase_reverse_generation


def routine_reverse_generation(p, tdp, model, id_epoch):

  nb_channels       = 1 if p.RGB_OR_GRAYSCALE == 'grayscale' else 3
  batch_size        = 1
  shape             = (batch_size, nb_channels, p.IMA_SIZE, p.IMA_SIZE)
  id_image_in_batch = 0
  nb_images         = 4
  generated_images  = []

  with torch.no_grad():

    for id_image in tqdm(range(nb_images), desc = 'Reverse generation', total = nb_images):
      img = torch.randn(shape, device = p.DEVICE)

      for id_timestep in reversed(range(0, p.NB_TIMESTEPS)):
        img = tdp.p_sample(
          model   = model,
          x       = img,
          t       = torch.full((p.BATCH_SIZE, ), id_timestep, device = p.DEVICE, dtype = torch.long),
          t_index = id_timestep)
      generated_images.append(trafo_tensor_to_pil(img, id_image_in_batch))

  showcase_reverse_generation(
    p                = p,
    generated_images = generated_images,
    nb_images        = nb_images,
    id_epoch         = id_epoch)
