import os
import matplotlib.pyplot as plt

from package_utils.trafo_pil_to_and_from_tensor import trafo_tensor_to_pil
from package_utils.nice_colorbar                import nice_colorbar

# FIXME --> Input parameters are too messy
def showcase_image(
  batch_images_clean,
  batch_images_noisy,
  batch_target_noise,
  batch_predicted_noise,
  betas,
  t,
  variance_schedule,
  id_epoch,
  id_batch,
  batch_names,
  trn_val_tst,
  results_path,
  rgb_or_grayscale):

  if id_batch == 0:

    id_ima_in_batch = 0

    # FIXME
    timestep = t[id_ima_in_batch].detach().cpu()

    result_name_suptitle = 'Set: {}, Epoch: {}, Batch: {}, Name: {}'.format(
      trn_val_tst,
      id_epoch,
      id_batch,
      batch_names[0])

    result_name_file = result_name_suptitle.replace(': ', '_').replace(', ', '_') + '.png'

    fig, ax = plt.subplots(3, 2)

    fig.set_dpi(300)
    fig.set_size_inches(8, 12, forward = True)

    plt.suptitle(result_name_suptitle)

    [axx.set_axis_off() for axx in ax.ravel()]

    axx = ax[0, 0]
    fig.sca(axx)
    axx.set_title(r'Original image:  $x_0$')
    im = axx.imshow(
      trafo_tensor_to_pil(batch_images_clean, id_ima_in_batch), vmin = 0, vmax = 255, cmap = 'gray')
    if rgb_or_grayscale == 'grayscale':
      nice_colorbar(im, axx)

    # FIXME --> In the original paper by Ho, "\epsilon_\theta" is function of "x(t)", ...
    # ...here the noise is always random and therefore independant of "x(t)"
    # "sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise"
    axx = ax[1, 0]
    fig.sca(axx)
    axx.set_title(r'Noisy image: $x_t = \sqrt{\overline{\alpha_t}}x_0 + \sqrt{1-\overline{\alpha_t}}\epsilon_\theta$')
    im = axx.imshow(
      trafo_tensor_to_pil(batch_images_noisy, id_ima_in_batch), vmin = 0, vmax = 255, cmap = 'gray')
    if rgb_or_grayscale == 'grayscale':
      nice_colorbar(im, axx)

    axx = ax[0, 1]
    fig.sca(axx)
    axx.set_title(r'Target noise: $\epsilon_\theta$')
    im = axx.imshow(
      trafo_tensor_to_pil(batch_target_noise, id_ima_in_batch), vmin = 0, vmax = 255, cmap = 'gray')
    if rgb_or_grayscale == 'grayscale':
      nice_colorbar(im, axx)

    axx = ax[1, 1]
    fig.sca(axx)
    axx.set_title(r'Predicted noise: $\Phi(x_t)$')
    fig.sca(axx)
    im = axx.imshow(
      trafo_tensor_to_pil(batch_predicted_noise, id_ima_in_batch), vmin = 0, vmax = 255, cmap = 'gray')
    if rgb_or_grayscale == 'grayscale':
      nice_colorbar(im, axx)

    axx = ax[2, 0]
    fig.sca(axx)
    axx.set_axis_on()
    axx.set_title(r'$\beta$ schedule ({}, t={})'.format(variance_schedule, timestep))
    plt.plot(betas)
    plt.plot(timestep, betas[timestep], '*')
    plt.xlabel('Timesteps')
    plt.ylabel(r'$\beta$')

    axx = ax[2, 1]
    fig.sca(axx)
    axx.set_title(r'Pixel-wise absolute error: $|\epsilon_\theta - \Phi(x_t)|$')
    im = axx.imshow(
      trafo_tensor_to_pil(abs(batch_target_noise - batch_predicted_noise), id_ima_in_batch),
      vmin = 0, vmax = 255, cmap = 'gray')
    if rgb_or_grayscale == 'grayscale':
      nice_colorbar(im, axx)

    fig.savefig(os.path.join(results_path, result_name_file), bbox_inches = 'tight')
    plt.close()
