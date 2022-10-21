import torch


def get_optimizer_and_scheduler(p, model):

  # Optimizer
  if p.OPTIMIZER_NICKNAME == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr = p.LEARNING_RATE, momentum = p.MOMENTUM)
  elif p.OPTIMIZER_NICKNAME == 'ADAM':
    optimizer = torch.optim.Adam(model.parameters(), lr = p.LEARNING_RATE)
  else:
    raise NotImplementedEror()

  # Scheduler
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, p.SCHEDULER_STEP_SIZE, p.SCHEDULER_GAMMA)

  return optimizer, scheduler
