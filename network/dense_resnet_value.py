"""Dense Resnet Value Network."""
import gin
import torch
import torch.nn as nn


@gin.configurable
class DenseResnetValue(nn.Module):
  """Dense Resnet layer."""

  def __init__(self, input_dim, width=512, num_blocks=2):
    super().__init__()
    self.dense0 = nn.Linear(input_dim, width)
    self.blocks = [ResNetDenseBlock(width) for _ in range(num_blocks)]
    self.blocks = nn.Sequential(*self.blocks)
    self.dense1 = nn.Linear(width, 1)

  def forward(self, x):
    x = self.dense0(x)
    x = self.blocks(x)
    x = self.dense1(x)
    return x


class ResNetDenseBlock(nn.Module):
  """Dense resnet block."""

  def __init__(self, width):
    super().__init__()
    self.dense0 = nn.Linear(width, width // 4)
    self.dense1 = nn.Linear(width // 4, width // 4)
    self.dense2 = nn.Linear(width // 4, width)
    # self.dense3 = nn.Linear(width, width)

    self.activation0 = nn.ReLU()
    self.activation1 = nn.ReLU()
    self.activation2 = nn.ReLU()
    # self.activation3 = nn.ReLU()

  def forward(self, x):
    y = self.dense0(self.activation0(x))
    y = self.dense1(self.activation1(y))
    y = self.dense2(self.activation2(y))
    # in definition we have enforce same shape
    # if x.shape != y.shape:
    #   x = self.dense3(self.activation3(x))
    return x + y

if __name__ == "__main__":
    input_dim = 1024
    input = torch.rand(input_dim)
    print(input.shape)
    model = DenseResnetValue(input_dim)
    print(model)
    out = torch.squeeze(model(input))
    print(out)