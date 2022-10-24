import os
from shutil import copyfile


def backup_parameters(backup_parameters_path):

  # 1. Copy the Git-ignored file "set_parameters.py" into the Git-tracked file "get_parameters.py" ...
  copyfile(
    os.path.join('package_parameters', 'set_parameters.py'),
    os.path.join('package_parameters', 'get_parameters.py'))

  # ... and within "get_parameters.py", modify the function name from "set_parameters(...)" to "get_parameters(...)"
  fid = open(os.path.join('package_parameters', 'get_parameters.py'), 'rt')
  data = fid.read()
  data = data.replace('set_parameters()', 'get_parameters()')
  fid.close()
  fid = open(os.path.join('package_parameters', 'get_parameters.py'), 'wt')
  fid.write(data)
  fid.close()

  # 2. Make a Git-ignored copy of the parameter files needed to re-run the experiment, and store them next to the results
  copyfile(
    os.path.join('package_parameters', 'get_parameters.py'),
    os.path.join(backup_parameters_path, 'get_parameters.py'))
  copyfile(
    os.path.join('package_parameters', 'parameters.py'),
    os.path.join(backup_parameters_path, 'parameters.py'))
  copyfile(
    os.path.join('package_parameters', '__init__.py'),
    os.path.join(backup_parameters_path, '__init__.py'))
