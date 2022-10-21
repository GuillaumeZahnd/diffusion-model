import torch
import os
import numpy as np
from PIL import Image
from package_utils.trafo_pil_to_and_from_tensor import trafo_pil_to_tensor


class Dataset(torch.utils.data.Dataset):

  # ----------------------------------------------------------------
  def __init__(self, dataset_path, ima_extension, ima_size):
    self.list_of_all_files, self.list_of_all_subfolders = \
      list_all_files_in_all_subfolders(dataset_path, ima_extension)
    self.ima_extension = ima_extension
    self.ima_size = ima_size


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
