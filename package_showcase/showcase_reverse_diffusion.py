import matplotlib.pyplot as plt
import math
import os


def showcase_reverse_diffusion(images_over_timesteps, p, id_epoch):

  # TODO --> Adaptative number of panels w.r.t. the number of timesteps
  fig, ax = plt.subplots(10, 20)

  fig.suptitle('Epoch: {}'.format(id_epoch))

  fig.set_dpi(300)
  fig.set_size_inches(20*2, 10*2, forward = True)

  [axx.set_axis_off() for axx in ax.ravel()]

  for idx in range(p.NB_TIMESTEPS):
    id_x = idx % 20
    id_y = math.floor(idx / 20)

    axx = ax[id_y, id_x]
    fig.sca(axx)
    axx.set_title('t={}'.format(idx))
    im = axx.imshow(images_over_timesteps[idx], vmin = 0, vmax = 255, cmap = 'gray')

  file_name = 'diffusion_{:002d}.png'.format(id_epoch)
  fig.savefig(os.path.join(p.RESULTS_IMAGES_EPOCHS, file_name), bbox_inches = 'tight')
  plt.close()
