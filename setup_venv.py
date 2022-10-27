import os
import re


if __name__ == '__main__':

  # Create the virtual environment directory ".venv"
  virtual_environment_dir = '.venv'
  if not os.path.exists(virtual_environment_dir):
    os.mkdir(virtual_environment_dir)

  # Attempt to retrieve the Python version from the Pipfile
  with open('Pipfile') as f:
    content = f.read()
    pattern = '(?:python_version.*)'
    match = re.search(pattern, content)
  if not match:
    raise ValueError('The Python version is not set in the Pipfile')

  # Remove undesired characters from the retrieved string
  python_version = match[0]
  python_version = python_version.replace('python_version', '') # Label
  python_version = python_version.replace('=', '')              # Equal symbol
  python_version = python_version.replace('\'', '')             # Apostrophe
  python_version = python_version.replace('\"', '')             # Double apostrophe
  python_version = python_version.replace(' ', '')              # Space
  print('Python version: {}'.format(python_version))

  # Install venv defined in pipenv
  stream = os.popen('pipenv install -d --python {}'.format(python_version))
  output = stream.read()
  output
