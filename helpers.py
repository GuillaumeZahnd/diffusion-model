def exists(x):
  return x is not None


def default(val, d):
  if exists(val):
    return val
  return d() if isfunction(d) else d


class Residual(nn.Module):
  def __init__(self, fn):
    super().__init__()
    self.fn = fn

  def forward(self, x, *args, **kwargs):
    return self.fn(x, *args, **kwargs) + x


def Upsample(dim):
  return nn.ConvTranspose2d(dim, dim, 4, 2, 1)


def Downsample(dim):
  return nn.Conv2d(
    in_channels  = dim,
    out_channels = dim,
    kernel_size  = 4,
    stride       = 2,
    padding      = 1)
