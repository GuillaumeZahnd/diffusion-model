import torch.nn as nn


# NOTE: Still no consensus today whether to apply normalization pre- or post-attention in Transformers
class PreNormalization(nn.Module):

  def __init__(self, dim, fn):
    super().__init__()
    self.fn = fn
    # NOTE: Group normalization is an alternative to batch normalization (https://arxiv.org/abs/1803.08494)
    self.norm = nn.GroupNorm(1, dim)

  def forward(self, x):
    return self.fn(self.norm(x))
