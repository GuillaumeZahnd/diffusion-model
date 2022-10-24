from package_loggers.print_line_separator import print_line_separator


def print_all_experiment_parameters(p):
  p_attributes = vars(p)
  print('\n'.join("%s: %s" % item for item in p_attributes.items()))
  print_line_separator()
