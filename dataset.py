import torch
from torchvision import transforms
import os
import numpy as np
from PIL import Image


class Dataset(torch.utils.data.Dataset):

  # ----------------------------------------------------------------
  def __init__(self, p):
    self.list_of_all_files, self.list_of_all_subfolders = \
      list_all_files_in_all_subfolders(p.DATASET_PATH, p.IMAGE_EXTENSION)
    self.ima_extension = p.IMAGE_EXTENSION
    self.ima_size = p.IMAGE_SIZE


  # ----------------------------------------------------------------
  def __len__(self):
    return len(self.list_of_all_files)


  # ----------------------------------------------------------------
  def __getitem__(self, idx):
    return self.list_of_all_files[idx], trafo_pil_to_tensor(
      ima_pil = load_ima(
        ima_path      = self.list_of_all_subfolders[idx],
        ima_name      = self.list_of_all_files[idx],
        ima_extension = self.ima_extension),
      ima_size = self.ima_size)


# ----------------------------------------------------------------
def list_all_files_in_all_subfolders(root_folder, ima_extension):
  list_of_all_files = []
  list_of_all_subfolders = []
  for all_subfolders, _, all_files in os.walk(root_folder):
    for file_name in all_files:
      if file_name.endswith(ima_extension):                            # <-- Custom filter
        list_of_all_files.append(file_name.replace(ima_extension, '')) # <-- We mark down the name of this file...
        list_of_all_subfolders.append(all_subfolders)                  # <-- ...which is located at this absolute path
  return list_of_all_files, list_of_all_subfolders


# ----------------------------------------------------------------
def load_ima(ima_path, ima_name, ima_extension):
  return Image.open(os.path.join(ima_path, ima_name + ima_extension))


# ----------------------------------------------------------------
def trafo_pil_to_tensor(ima_pil, ima_size):
  trafo = transforms.Compose([
    transforms.Resize(ima_size),
    transforms.CenterCrop(ima_size),
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
