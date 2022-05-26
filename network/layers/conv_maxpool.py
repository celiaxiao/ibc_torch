"""Simple 4-layer conv + maxpool CNN."""
import torch
import torch.nn as nn

# Important: unlike tensorflow assume channel last, pytorch requires channel first (N, C, W, H)
# Important: output shape is (N,C,1,1) or (C,1,1), make sure to squeeze afterwards
def get_conv_maxpool(nchannels):
    """Instantiates simple cnn architecture."""
    hidden_channels = [32, 64, 128, 256]
    kernel_size = 3
    maxpool_ker_size = (2, 2)
    model = nn.Sequential(
        nn.Conv2d(in_channels=nchannels,out_channels=hidden_channels[0], kernel_size=kernel_size, padding='same',),
        nn.ReLU(),
        nn.MaxPool2d(maxpool_ker_size),
        nn.Conv2d(in_channels=hidden_channels[0],out_channels=hidden_channels[1], kernel_size=kernel_size, padding='same',),
        nn.ReLU(),
        nn.MaxPool2d(maxpool_ker_size),
        nn.Conv2d(in_channels=hidden_channels[1],out_channels=hidden_channels[2], kernel_size=kernel_size, padding='same',),
        nn.ReLU(),
        nn.MaxPool2d(maxpool_ker_size),
        nn.Conv2d(in_channels=hidden_channels[2],out_channels=hidden_channels[3], kernel_size=kernel_size, padding='same',),
        nn.ReLU(),
        nn.MaxPool2d(maxpool_ker_size),
        nn.AdaptiveAvgPool2d((1,1))
    )
    return model

if __name__ == "__main__":
    N = 8
    C = 16
    H = 96
    W = 128
    input = torch.rand((N, C, W, H))
    print(input.shape)
    model = get_conv_maxpool( C)
    print(model)
    out = torch.squeeze(model(input))
    print(out.shape) # [N, 256]