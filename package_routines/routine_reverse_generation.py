import torch
from tqdm import tqdm

from package_utils.trafo_pil_to_and_from_tensor   import trafo_tensor_to_pil
from package_showcase.showcase_reverse_generation import showcase_reverse_generation


def routine_reverse_generation(p, tdp, model, id_epoch = None):

  batch_size       = 9 # Re-definition of the batch size (TODO: Use eiher this solution, or "batch_size=1" with a For loop over "nb_images")
  nb_channels      = 1 if p.RGB_OR_GRAYSCALE == 'grayscale' else 3
  shape            = (batch_size, nb_channels, p.IMA_SIZE, p.IMA_SIZE)
  generated_images = []

  with torch.no_grad():

    img = torch.randn(shape, device = p.DEVICE)

    for id_timestep in tqdm(reversed(range(0, p.NB_TIMESTEPS)), desc = 'Reverse generation', total = p.NB_TIMESTEPS):
      img = tdp.p_sample(
        model   = model,
        x       = img,
        t       = torch.full((batch_size, ), id_timestep, device = p.DEVICE, dtype = torch.long),
        t_index = id_timestep)

  for id_image_in_batch in range(batch_size):
    generated_images.append(trafo_tensor_to_pil(img, id_image_in_batch))

  showcase_reverse_generation(
    p                = p,
    generated_images = generated_images,
    nb_images        = batch_size,
    id_epoch         = id_epoch)
