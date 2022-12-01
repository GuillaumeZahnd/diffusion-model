import matplotlib.pyplot as plt
import math
import os


def showcase_reverse_diffusion(img_through_timesteps, p, id_epoch):

  # Diffusion is presented in a X-by-Y grid, where "nb_panels" images among "p.NB_TIMESTEPS" are regularly displayed
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

    # Order items similarly as during the diffusion process: first is pure noise (t=T), last is generated image (t=0)
    id_timestep_reverse = int(p.NB_TIMESTEPS - id_panel * stride)

    id_panel_x = id_panel % nb_panels_x
    id_panel_y = math.floor(id_panel / nb_panels_x)

    axx = ax[id_panel_y, id_panel_x]
    fig.sca(axx)
    axx.set_title('t={}'.format(id_timestep_reverse))
    im = axx.imshow(img_through_timesteps[id_timestep_reverse], vmin = 0, vmax = 255, cmap = 'gray')

  file_name = 'diffusion_{:03d}.png'.format(id_epoch)
  fig.savefig(os.path.join(p.RESULTS_IMAGES_DIFFUSION, file_name), bbox_inches = 'tight')
  plt.close()
