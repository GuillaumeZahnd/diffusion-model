import torch
from tqdm import tqdm

from package_utils.trafo_pil_to_and_from_tensor  import trafo_tensor_to_pil
from package_showcase.showcase_reverse_diffusion import showcase_reverse_diffusion


def routine_reverse_loop(p, tdp, model, id_epoch):

  with torch.no_grad():

    nb_channels = 1 if p.RGB_OR_GRAYSCALE == 'grayscale' else 3
    shape = (p.BATCH_SIZE, nb_channels, p.IMA_SIZE, p.IMA_SIZE)

    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device = p.DEVICE)
    imgs = []

    id_image_in_batch = 0

    for id_timestep in tqdm(reversed(range(0, p.NB_TIMESTEPS)), desc = 'Reverse diffusion', total = p.NB_TIMESTEPS):
      img = tdp.p_sample(
        model = model,
        x = img,
        t = torch.full((p.BATCH_SIZE, ), id_timestep, device = p.DEVICE, dtype = torch.long),
        t_index = id_timestep)
      imgs.append(trafo_tensor_to_pil(img, id_image_in_batch))

  showcase_reverse_diffusion(imgs, p, id_epoch)
