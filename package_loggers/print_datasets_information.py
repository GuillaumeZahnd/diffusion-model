from package_loggers.print_line_separator import print_line_separator


def print_datasets_information(loader_trn, loader_val, batch_size):
  print('Training set:\t number of samples = {}\t number of batches = {}\t batch size = {}'.format(
    len(loader_trn.dataset), len(loader_trn), batch_size))
  print('Validation set:\t number of samples = {}\t number of batches = {}\t batch size = {}'.format(
    len(loader_val.dataset), len(loader_val), batch_size))
  print_line_separator()
