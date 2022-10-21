import os
from torchvision import transforms
from PIL import Image
import numpy as np


# ----------------------------------------------------------------
def load_ima():
  dataset_path    = '/media/guillaume/f0bb3659-b50a-4aac-b559-0953c567b645/afhq/'
  dataset_split   = 'train'
  animal_category = 'wild'
  ima_path        = os.path.join(dataset_path, dataset_split, animal_category)
  ima_name        = 'flickr_wild_000002.jpg'
  return Image.open(os.path.join(ima_path, ima_name))


# ----------------------------------------------------------------
def trafo_pil_to_tensor(ima_pil):
  image_size = 512
  trafo = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(), # From PIL Image (H x W x C) in range [0, 255] to torch.FloatTensor (C x H x W) in range [0.0, 1.0]
    transforms.Lambda(lambda t: (t * 2) - 1),
    ])
  return trafo(ima_pil).unsqueeze(0)


# ----------------------------------------------------------------
def trafo_tensor_to_pil(ima_tensor):
  trafo = transforms.Compose([
    transforms.Lambda(lambda t: (t + 1) / 2),
    transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC # TODO batch
    transforms.Lambda(lambda t: t * 255.),
    transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
    transforms.ToPILImage(),
    ])
  return trafo(ima_tensor.squeeze())
