import torch.nn as nn


class PreNormalization(nn.Module): # <-- Still no consensus today whether to apply normalization pre- or post- attention in Transformers

  def __init__(self, dim, fn):
    super().__init__()
    self.fn = fn
    self.norm = nn.GroupNorm(1, dim) # <-- An alternative to batch normalization (https://arxiv.org/abs/1803.08494)

  def forward(self, x):
    return self.fn(self.norm(x))
