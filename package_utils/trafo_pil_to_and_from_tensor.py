import numpy as np
from torchvision import transforms


# ----------------------------------------------------------------
def trafo_pil_to_tensor(ima_pil, ima_size):
  trafo = transforms.Compose([
    transforms.Resize(ima_size),
    transforms.CenterCrop(ima_size),
    transforms.RandomHorizontalFlip(p = 0.5),
    transforms.ToTensor(), # From PIL Image (H x W x C) in range [0, 255] to torch.FloatTensor (C x H x W) in range [0.0, 1.0]
    transforms.Lambda(lambda t: (t * 2) - 1)])
  return trafo(ima_pil)


# ----------------------------------------------------------------
def trafo_tensor_to_pil(ima_tensor, id_ima_in_batch):
  trafo = transforms.Compose([
    transforms.Lambda(lambda t: (t + 1) / 2),
    transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
    transforms.Lambda(lambda t: t * 255.),
    transforms.Lambda(lambda t: t.cpu()),
    transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
    transforms.ToPILImage()])
  return trafo(ima_tensor[id_ima_in_batch,:,:,:].detach())
