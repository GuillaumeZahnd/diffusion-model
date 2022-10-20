from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage import io
import matplotlib.pyplot as plt
import os
import numpy as np


# ----------------------------------------------------------------
def nice_colorbar(im, axx):
  divider = make_axes_locatable(axx)
  cax = divider.append_axes('right', size = '5%', pad = 0.05)
  cbar = plt.colorbar(im, cax=cax)
  return cbar


# ----------------------------------------------------------------
if __name__ == '__main__':

  dataset_path    = '/media/guillaume/f0bb3659-b50a-4aac-b559-0953c567b645/afhq/'
  dataset_split   = 'train'
  animal_category = 'wild'

  ima_path = os.path.join(dataset_path, dataset_split, animal_category)

  ima = io.imread(ima_path + '/flickr_wild_000002.jpg').astype('float32')

  ima = ima / np.max(ima)
  ima = 2 * (ima - 0.5)

  print(ima.shape)


  mu = 0
  sigma = 0.5
  gaussian_noise = np.random.normal(loc = mu, scale = sigma, size = ima.shape)


  fig, ax = plt.subplots(3, 1)

  axx = ax[0]
  fig.sca(axx)
  im = axx.imshow(ima)
  plt.xticks([])
  plt.yticks([])
  nice_colorbar(im, axx)

  axx = ax[1]
  fig.sca(axx)
  im = axx.imshow(gaussian_noise)
  plt.xticks([])
  plt.yticks([])
  nice_colorbar(im, axx)

  axx = ax[2]
  fig.sca(axx)
  im = axx.imshow(ima + gaussian_noise)
  plt.xticks([])
  plt.yticks([])
  nice_colorbar(im, axx)

  plt.show()
