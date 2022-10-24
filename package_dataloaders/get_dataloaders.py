import torch
from package_dataloaders.dataset import Dataset


def get_dataloaders(p):

  # Dataloader for the training ("trn") dataset
  params_dataloader_trn = {
    'batch_size' : p.BATCH_SIZE,
    'shuffle'    : True,
    'num_workers': 4,
    'drop_last'  : False}
  loader_trn = torch.utils.data.DataLoader(
    dataset = Dataset(
      dataset_path     = p.DATASET_TRN_PATH,
      nb_samples_limit = p.NB_SAMPLES_LIMIT,
      ima_extension    = p.IMA_EXTENSION,
      ima_size         = p.IMA_SIZE),
    **params_dataloader_trn)

  # Dataloader for the validation ("val") dataset
  params_dataloader_val = {
    'batch_size' : p.BATCH_SIZE,
    'shuffle'    : False,
    'num_workers': 4,
    'drop_last'  : False}
  loader_val = torch.utils.data.DataLoader(
    dataset = Dataset(
      dataset_path     = p.DATASET_VAL_PATH,
      nb_samples_limit = p.NB_SAMPLES_LIMIT,
      ima_extension    = p.IMA_EXTENSION,
      ima_size         = p.IMA_SIZE),
    **params_dataloader_val)

  print('Training set:\t number of samples = {}\t number of batches = {}\t batch size = {}'.format(
    len(loader_trn.dataset), len(loader_trn), p.BATCH_SIZE))
  print('Validation set:\t number of samples = {}\t number of batches = {}\t batch size = {}'.format(
    len(loader_val.dataset), len(loader_val), p.BATCH_SIZE))

  return loader_trn, loader_val
