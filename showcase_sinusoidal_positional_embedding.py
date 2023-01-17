import torch
import numpy as np
import matplotlib.pyplot as plt

from package_parameters.set_parameters            import set_parameters
from package_model.sinusoidal_position_embeddings import SinusoidalPositionEmbeddings


if __name__ == '__main__':

  p = set_parameters()

  spe = SinusoidalPositionEmbeddings(dim = p.IMA_SIZE)
  time = torch.from_numpy(np.arange(512))
  embeddings = spe(time)
  print(embeddings.shape)

  fig, ax = plt.subplots(1, 1)
  fig.set_dpi(300)
  fig.set_size_inches(8, 32, forward = True)

  plt.imshow(embeddings)
  plt.colorbar()

  fig.savefig('sinusoidal_positional_embedding.png', bbox_inches = 'tight')
  plt.close()
