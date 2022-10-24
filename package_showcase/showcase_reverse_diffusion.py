import matplotlib.pyplot as plt
import math
import os


def showcase_reverse_diffusion(images_over_timesteps, p, id_epoch):

  # TODO --> Adaptative number of panels w.r.t. the number of timesteps
  fig, ax = plt.subplots(10, 20)

  fig.suptitle(id_epoch)

  fig.set_dpi(300)
  fig.set_size_inches(20*3, 10*3, forward = True)

  [axx.set_axis_off() for axx in ax.ravel()]

  for idx in range(p.NB_TIMESTEPS):
    id_x = idx % 20
    id_y = math.floor(idx / 20)
    ima = images_over_timesteps[idx]

    axx = ax[id_y, id_x]
    fig.sca(axx)
    axx.set_title(idx)
    im = axx.imshow(ima, vmin = 0, vmax = 255, cmap = 'gray')

  fig.savefig(os.path.join(p.RESULTS_IMAGES_EPOCHS, str(id_epoch) + '.png'), bbox_inches = 'tight')
  plt.close()
