import os
import math
import torch
import random
import numpy as np
from PIL import Image
from package_utils.trafo_pil_to_and_from_tensor import trafo_pil_to_tensor


class Dataset(torch.utils.data.Dataset):

  # ----------------------------------------------------------------
  def __init__(self, dataset_path, nb_samples_limit, ima_extension, ima_size, rgb_or_grayscale):

    self.ima_extension    = ima_extension
    self.ima_size         = ima_size
    self.rgb_or_grayscale = rgb_or_grayscale

    self.list_of_all_files, self.list_of_all_subfolders = list_all_matching_files_in_all_matching_subfolders(
      root_folder   = dataset_path,
      ima_extension = ima_extension)

    if nb_samples_limit < len(self.list_of_all_files):
      selected_samples_idx = np.arange(nb_samples_limit)
      np.random.shuffle(selected_samples_idx)
      self.list_of_all_files      = [self.list_of_all_files[idx] for idx in selected_samples_idx]
      self.list_of_all_subfolders = [self.list_of_all_subfolders[idx] for idx in selected_samples_idx]


  # ----------------------------------------------------------------
  def __len__(self):
    return len(self.list_of_all_files)


  # ----------------------------------------------------------------
  def __getitem__(self, idx):
    # TODO --> Add option to convert RGB to Gray according to "parameters.py"
    return self.list_of_all_files[idx], trafo_pil_to_tensor(
      ima_pil = load_ima(
        ima_path         = self.list_of_all_subfolders[idx],
        ima_name         = self.list_of_all_files[idx],
        ima_extension    = self.ima_extension,
        rgb_or_grayscale = self.rgb_or_grayscale),
      ima_size = self.ima_size)


# ----------------------------------------------------------------
def list_all_matching_files_in_all_matching_subfolders(root_folder, ima_extension):
  list_of_all_files = []
  list_of_all_subfolders = []
  for all_subfolders, _, all_files in os.walk(root_folder):
    for file_name in all_files:
      # This is currently the only "matching" condition, it can be adapted according to specific needs
      if file_name.endswith(ima_extension):
        list_of_all_files.append(file_name.replace(ima_extension, ''))
        list_of_all_subfolders.append(all_subfolders)
  return list_of_all_files, list_of_all_subfolders


# ----------------------------------------------------------------
def load_ima(ima_path, ima_name, ima_extension, rgb_or_grayscale):
  ima = Image.open(os.path.join(ima_path, ima_name + ima_extension))
  if rgb_or_grayscale == 'grayscale':
    return ima.convert('L')
  else:
    return ima
