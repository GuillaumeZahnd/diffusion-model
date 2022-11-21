import numpy as np
from torchvision import transforms


# ----------------------------------------------------------------
def trafo_pil_to_tensor(ima_pil, ima_size):
  trafo = transforms.Compose([
    transforms.CenterCrop(min(ima_pil.width, ima_pil.height)), # Squarify the image across the shortest side
    transforms.Resize(ima_size),                               # Resize the image to the specified side-length
    transforms.RandomHorizontalFlip(p = 0.5),                  # Apply horizontal mirroring (50% chance)
    transforms.ToTensor(),                                     # From HWC in range [0, 255] to CHW in range [0, 1]
    transforms.Lambda(lambda t: (t * 2) - 1)                   # Set the range to [-1, +1]
    ])
  return trafo(ima_pil)


# ----------------------------------------------------------------
def trafo_tensor_to_pil(ima_tensor, id_ima_in_batch):
  trafo = transforms.Compose([
    transforms.Lambda(lambda t: (t + 1) / 2),                # Set the range to [0, 1]
    transforms.Lambda(lambda t: t * 255.),                   # Set the range to [0, 255]
    transforms.Lambda(lambda t: t.permute(1, 2, 0)),         # CHW to HWC
    transforms.Lambda(lambda t: t.cpu()),                    # Gather to CPU from GPU
    transforms.Lambda(lambda t: t.numpy().astype(np.uint8)), # Transform from Tensor to Numpy array
    transforms.ToPILImage()                                  # Convert to PIL
    ])
  return trafo(ima_tensor[id_ima_in_batch,:,:,:].detach())
