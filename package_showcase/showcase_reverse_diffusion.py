import matplotlib.pyplot as plt
import math
import os


def showcase_reverse_diffusion(img_through_timesteps, p, id_epoch):

  # Diffusion is presented in a 10-by-10 grid
  nb_panels_x = 10
  nb_panels_y = 10
  nb_panels   = nb_panels_x * nb_panels_y
  stride      = p.NB_TIMESTEPS / (nb_panels -1)

  fig, ax = plt.subplots(nb_panels_x, nb_panels_y)
  fig.set_dpi(300)
  fig.set_size_inches(20, 20, forward = True)
  fig.suptitle('Epoch: {}'.format(id_epoch))

  [axx.set_axis_off() for axx in ax.ravel()]

  for id_panel in range(nb_panels):

    id_timestep         = int(id_panel * stride)       # Here, the first item corresponds to t=0 and the last to t=T
    id_timestep_reverse = p.NB_TIMESTEPS - id_timestep # Here, the items are ordered as during the reverse diffusion

    id_panel_x = id_panel % nb_panels_x
    id_panel_y = math.floor(id_panel / nb_panels_x)

    axx = ax[id_panel_y, id_panel_x]
    fig.sca(axx)
    axx.set_title('t={}'.format(id_timestep_reverse))
    im = axx.imshow(img_through_timesteps[id_timestep_reverse], vmin = 0, vmax = 255, cmap = 'gray')

  file_name = 'diffusion_{:03d}.png'.format(id_epoch)
  fig.savefig(os.path.join(p.RESULTS_IMAGES_EPOCHS, file_name), bbox_inches = 'tight')
  plt.close()
