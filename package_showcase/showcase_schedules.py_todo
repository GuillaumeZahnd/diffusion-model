import matplotlib.pyplot as plt
from schedule import define_schedule


if __name__ == '__main__':

  nb_timesteps = 200

  betas_cosine    = define_schedule(schedule_strategy = 'cosine', nb_timesteps = nb_timesteps)
  betas_linear    = define_schedule(schedule_strategy = 'linear', nb_timesteps = nb_timesteps)
  betas_quadratic = define_schedule(schedule_strategy = 'quadratic', nb_timesteps = nb_timesteps)
  betas_sigmoid   = define_schedule(schedule_strategy = 'sigmoid', nb_timesteps = nb_timesteps)

  fig, ax = plt.subplots(2, 1)
  fig.set_dpi(300)
  fig.set_size_inches(8, 16, forward = True)

  axx = ax[0]
  fig.sca(axx)
  plt.plot(betas_cosine, label = 'cosine')
  plt.plot(betas_linear, label = 'linear')
  plt.plot(betas_quadratic, label = 'quadratic')
  plt.plot(betas_sigmoid, label = 'sigmoid')
  plt.xlabel('Timesteps')
  plt.ylabel('beta')
  plt.legend()

  axx = ax[1]
  fig.sca(axx)
  plt.plot(betas_linear, label = 'linear')
  plt.plot(betas_quadratic, label = 'quadratic')
  plt.plot(betas_sigmoid, label = 'sigmoid')
  plt.xlabel('Timesteps')
  plt.ylabel('beta')
  plt.legend()

  fig.savefig('schedules.png', bbox_inches = 'tight')
  plt.close()
