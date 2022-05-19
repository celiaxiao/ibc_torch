from agents.utils import tile_batch
from network import conv_maxpool, dense_resnet_value
import torch
import torch.nn as nn
import gin
from network import image_prepro
def get_encoder_network(encoder_network, channels):
    return conv_maxpool.get_conv_maxpool(channels)


def get_value_network(value_network, input_dim):
  if value_network == 'DenseResnetValue':
    return dense_resnet_value.DenseResnetValue(input_dim)
  else:
    raise ValueError('Unsupported value_network %s' % value_network)

@gin.configurable
class PixelEBM(nn.Module):
  """Late fusion PixelEBM."""

  def __init__(self,
               obs_spec: dict,
               action_dim: int,
               encoder_network: str,
               value_network: str,
               N: int,
               sequence_length=2,
               target_height=90,
               target_width=120,
               name='PixelEBM'):
    super().__init__()
    # We stack all images and coord-conv.
    num_channels = (3 * sequence_length)
    self._encoder = get_encoder_network(encoder_network,num_channels)
    self.target_height = target_height
    self.target_width = target_width
    # image post precess has shape [N, 256]
    # actions has shape [N, action_dim]
    input_dim = (256 + action_dim) 
    self._value = get_value_network(value_network, input_dim)

    rgb_shape = obs_spec['rgb'].shape
    self._static_height = rgb_shape[1]
    self._static_width = rgb_shape[2]
    self._static_channels = rgb_shape[3]

  def encode(self, obs):
    """Embeds images."""
    images = obs['rgb']
    # print("brfore process", images.shape)
    images = image_prepro.preprocess(images,
                                     target_height=self.target_height,
                                     target_width=self.target_width)
    # print("after process", images.shape)
    observation_encoding = self._encoder(images)
    return torch.squeeze(observation_encoding) # [batch_size, 256]

  def forward(self, inputs, observation_encoding=None):
    obs, act = inputs

    # If we pass in observation_encoding, we are doing late fusion.
    if observation_encoding is None:
      # Otherwise embed for the first time.
      observation_encoding = self.encode(obs)
      batch_size = obs['rgb'].shape[0]
      num_samples = act.shape[0] // batch_size
      observation_encoding = tile_batch(
          observation_encoding, num_samples)

    # print("after encoding", observation_encoding.shape)
    # Concat [obs, act].
    x = torch.concat([observation_encoding, act], -1)
    # x = torch.flatten(x)
    # Forward value network.
    x = self._value(x)

    # Squeeze extra dim.
    x = torch.squeeze(x, axis=-1)

    return x # single value tensor

