import torch
from package_dataloaders.dataset import Dataset
from package_loggers.print_datasets_information import print_datasets_information


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

  # Print information about the datasets
  print_datasets_information(loader_trn = loader_trn, loader_val = loader_val, batch_size = p.BATCH_SIZE)

  return loader_trn, loader_val
