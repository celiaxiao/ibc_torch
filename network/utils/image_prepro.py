import torch
from torchvision.transforms.functional import convert_image_dtype, resize
def stack_images_channelwise(obs, batch_size):
  # Use static shapes for hist, width, height, and channels since TPUs prefer
  # static shapes for some image ops. The batch size passed in may still be
  # dynamic.
  nc = obs.shape[1]
  nw = obs.shape[2]
  nh = obs.shape[3]
  nhist = obs.shape[4]
  obs = torch.reshape(obs, [batch_size, nc * nhist, nw, nh])
  return obs


def preprocess(images, target_height, target_width):
  """Converts to [0,1], stacks, resizes."""
  # Scale to [0, 1].
  images = convert_image_dtype(images, dtype=torch.float32)

  # Stack images channel-wise.
  batch_size = images.shape[0]
  images = stack_images_channelwise(images, batch_size)

  # Resize to target height and width.
  images = resize(images, [target_height, target_width])
  return images
