import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from package_model.helpers import exists, default, Upsample, Downsample
from package_model.residual_module                import ResidualModule
from package_model.pre_normalization              import PreNormalization
from package_model.building_blocks                import ResnetBlock
from package_model.building_blocks                import ConvNextBlock
from package_model.sinusoidal_position_embeddings import SinusoidalPositionEmbeddings
from package_model.attention                      import Attention
from package_model.attention                      import LinearAttention


class Unet(nn.Module):

  # ----------------------------------------------------------------
  def __init__(
    self,
    dim,
    channels,
    backbone,
    init_dim            = None,
    out_dim             = None,
    dim_mults           = (1, 2, 4, 8, 16),
    use_time_embeddings = True,
    resnet_block_groups = 8,
    convnext_mult       = 2):

    super().__init__()

    # determine dimensions
    self.channels = channels

    init_dim = default(init_dim, dim // 3 * 2)
    self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)

    dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
    in_out = list(zip(dims[:-1], dims[1:]))

    if backbone == 'CONVNEXT':
      block_klass = partial(ConvNextBlock, mult = convnext_mult)
    elif backbone == 'RESNET':
      block_klass = partial(ResnetBlock, groups = resnet_block_groups)
    else:
      raise NotImplementedError(backbone)

    # Time embeddings
    if use_time_embeddings:
      time_dim = dim * 4
      self.time_mlp = nn.Sequential(
        SinusoidalPositionEmbeddings(dim),
        nn.Linear(dim, time_dim),
        nn.GELU(),
        nn.Linear(time_dim, time_dim))
    else:
      time_dim = None
      self.time_mlp = None

    # Layers
    self.downs = nn.ModuleList([])
    self.ups = nn.ModuleList([])
    num_resolutions = len(in_out)

    for idx, (dim_in, dim_out) in enumerate(in_out):
      is_last = idx >= (num_resolutions - 1)
      self.downs.append(nn.ModuleList([
        block_klass(dim_in, dim_out, time_emb_dim=time_dim),
        block_klass(dim_out, dim_out, time_emb_dim=time_dim),
        ResidualModule(PreNormalization(dim_out, LinearAttention(dim_out))),
        Downsample(dim_out) if not is_last else nn.Identity()]))

    mid_dim = dims[-1]
    self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
    self.mid_attn = ResidualModule(PreNormalization(mid_dim, Attention(mid_dim)))
    self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

    for idx, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
      is_last = idx >= (num_resolutions - 1)
      self.ups.append(nn.ModuleList([
        block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
        ResidualModule(PreNormalization(dim_in, LinearAttention(dim_in))),
        Upsample(dim_in) if not is_last else nn.Identity()]))

    out_dim = default(out_dim, channels)

    self.final_conv = nn.Sequential(
      block_klass(dim, dim),
      nn.Conv2d(dim, out_dim, 1))


  # ----------------------------------------------------------------
  def forward(self, x, time):

    x = self.init_conv(x)
    t = self.time_mlp(time) if exists(self.time_mlp) else None
    h = []

    # Downsample
    for block1, block2, attn, downsample in self.downs:
      x = block1(x, t)
      x = block2(x, t)
      x = attn(x)
      h.append(x)
      x = downsample(x)

    # Bottleneck
    x = self.mid_block1(x, t)
    x = self.mid_attn(x)
    x = self.mid_block2(x, t)

    # Upsample
    for block1, block2, attn, upsample in self.ups:
      x = torch.cat((x, h.pop()), dim=1)
      x = block1(x, t)
      x = block2(x, t)
      x = attn(x)
      x = upsample(x)

    return self.final_conv(x)
