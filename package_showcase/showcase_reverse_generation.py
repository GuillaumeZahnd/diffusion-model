import matplotlib.pyplot as plt
import math
import os


def showcase_reverse_generation(generated_images, nb_images, p, id_epoch):

  fig, ax = plt.subplots(2, 2)

  fig.suptitle('Epoch: {}'.format(id_epoch))

  fig.set_dpi(300)
  fig.set_size_inches(12, 12, forward = True)

  [axx.set_axis_off() for axx in ax.ravel()]

  for idx in range(nb_images):
    id_x = idx % 2
    id_y = math.floor(idx / 2)

    axx = ax[id_y, id_x]
    fig.sca(axx)
    im = axx.imshow(generated_images[idx], vmin = 0, vmax = 255, cmap = 'gray')

  file_name = 'generation_{:002d}.png'.format(id_epoch)
  fig.savefig(os.path.join(p.RESULTS_IMAGES_EPOCHS, file_name), bbox_inches = 'tight')
  plt.close()
