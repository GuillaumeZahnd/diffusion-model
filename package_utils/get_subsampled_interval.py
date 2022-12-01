import numpy as np

# Given an interval of length "len_ini" populated by indices [0, 1, 2, ..., len_ini -1]
# Return a subsampled interval of length "len_sub" populated by regularly-spaced indices [0, ..., len_ini -1]
# Example:
# len_ini := 43 --> indices = [O, 1, 2, ... 43]
# len_sub := 7 --> indices = [0, 7, 14, 21, 28, 35, 42]

def get_subsampled_interval(len_ini, len_sub):
  stride = (len_ini -1) / (len_sub -1)
  return (stride * np.arange(len_sub)).astype(np.int32)
