import matplotlib.pyplot as plt

from package_parameters.set_parameters       import set_parameters
from package_diffusion.get_variance_schedule import get_variance_schedule


if __name__ == '__main__':

  p = set_parameters()

  p.VARIANCE_SCHEDULE = 'COSINE'
  betas_cosine    = get_variance_schedule(p)
  p.VARIANCE_SCHEDULE = 'LINEAR'
  betas_linear    = get_variance_schedule(p)
  p.VARIANCE_SCHEDULE = 'QUADRATIC'
  betas_quadratic = get_variance_schedule(p)
  p.VARIANCE_SCHEDULE = 'SIGMOID'
  betas_sigmoid   = get_variance_schedule(p)

  fig, ax = plt.subplots(1, 1)
  fig.set_dpi(300)
  fig.set_size_inches(8, 8, forward = True)

  fig.sca(ax)
  plt.plot(betas_linear, label = 'linear')
  plt.plot(betas_quadratic, label = 'quadratic')
  plt.plot(betas_sigmoid, label = 'sigmoid')
  plt.plot(betas_cosine, label = 'cosine')
  plt.xlabel('Timesteps')
  plt.ylabel('beta')
  plt.legend()

  fig.savefig('schedules.png', bbox_inches = 'tight')
  plt.close()
