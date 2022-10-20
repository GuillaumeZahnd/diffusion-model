import os
import torch
from medpy.io import load
from torchvision import transforms
from icecream import ic


# ----------------------------------------------------------------
if __name__ == '__main__':

  dataset_path    = '/media/guillaume/f0bb3659-b50a-4aac-b559-0953c567b645/afhq/'
  dataset_split   = 'train'
  animal_category = 'wild'
  ima_path        = os.path.join(dataset_path, dataset_split, animal_category)
  ima_name        = 'flickr_wild_000002.jpg'

  # Dataloader
  transform_input = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0], std=[1])])
  ima, _ = load(os.path.join(ima_path, ima_name))
  ima = transform_input(ima) # [512, 3, 512]
  ic(ima.shape)
