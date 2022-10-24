import torch
from tqdm import tqdm
from package_showcase.showcase_reverse_diffusion import showcase_reverse_diffusion
from package_utils.trafo_pil_to_and_from_tensor import trafo_tensor_to_pil


def routine_reverse_loop(p, tdp, model, id_epoch):

  with torch.no_grad():

    device = next(model.parameters()).device # TODO --> What is the difference with "p.DEVICE"?
    nb_channels = 3 # TODO --> Add as a parameters, incorporate in the dataloader
    shape = (p.BATCH_SIZE, nb_channels, p.IMA_SIZE, p.IMA_SIZE)

    bbb = shape[0] # FIXME
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device = p.DEVICE)
    imgs = []

    for id_timestep in tqdm(reversed(range(0, p.NB_TIMESTEPS)), desc = 'Reverse diffusion', total = p.NB_TIMESTEPS):
      img = tdp.p_sample(
        model,
        img,
        torch.full((bbb,), id_timestep, device = p.DEVICE, dtype = torch.long),
        id_timestep)
      imgs.append(trafo_tensor_to_pil(img[0,:,:,:])) # FIXME --> batch size

  showcase_reverse_diffusion(imgs, p, id_epoch)
