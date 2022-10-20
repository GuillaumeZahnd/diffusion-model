import torch
from dataset import Dataset


def get_dataloader(p):

  params_dataloader = {
    'batch_size' : p.BATCH_SIZE,
    'shuffle'    : True,
    'num_workers': 4,
    'drop_last'  : False}

  loader = torch.utils.data.DataLoader(dataset = Dataset(p), **params_dataloader)

  NB_BATCHES = len(loader)
  NB_SAMPLES = len(loader.dataset)

  print('Training set: number of samples = {}, number of batches = {}, batch size = {}'.format(
    NB_SAMPLES, NB_BATCHES, p.BATCH_SIZE))

  return loader, NB_BATCHES, NB_SAMPLES
