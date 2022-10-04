import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Yihe's implementation of pointnet layer.
Used for processing particles/pointcloud information before appending to agent info.
Ref: https://github.com/fxia22/pointnet.pytorch
'''

class pointNetLayer(nn.Module):
    def __init__(self, in_dim=[3,1024], out_dim=512, normalize=False):
        super().__init__()
        self.in_dim = in_dim
        self.normalize = normalize

        # global feature mlp
        self.conv1 = torch.nn.Conv1d(self.in_dim[0], 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 512, 1)

        if self.normalize:
            self.ln1 = torch.nn.LayerNorm([64, in_dim[1]])
            self.ln2 = torch.nn.LayerNorm([128, in_dim[1]])
            self.ln3 = torch.nn.LayerNorm([512, in_dim[1]])

        # project to desired output dim
        self.mlp_out = nn.Linear(512, out_dim)
        # zero_init_output
        nn.init.zeros_(self.mlp_out.bias)
        self.mlp_out.weight.data.copy_(0.01 * self.mlp_out.weight.data)

    def forward(self, x):
        # print(x.size())
        assert x.size()[-1] == self.in_dim[0] # make sure last dim = 3 or 6
        x = x.transpose(2, 1)
        
        x = F.relu(self.conv1(x))
        if self.normalize:
            x = self.ln1(x)
        x = F.relu(self.conv2(x))
        if self.normalize:
            x = self.ln2(x)
        x = self.conv3(x) # raw global feature
        if self.normalize:
            x = self.ln3(x)

        # print(x.shape)
        
        x = torch.max(x, 2, keepdim=True)[0] # [B, 1024, 1]
        # print(x.shape)
        x = x.view(-1, 512) # global feature, [B, 1024]
        # print(x.shape)
        x = self.mlp_out(x) # [B, 512]
        return x
    

if __name__ == "__main__":
    B = 234 # batch size
    N = 704 # num of points
    input = torch.rand((B, N, 3))
    print(input.shape)
    ptnet = pointNetLayer(in_dim=[3,704], normalize=True)
    print(ptnet)
    # out = torch.squeeze(ptnet(input))
    out = ptnet(input)
    print(out.shape) # [N, 256]
    print(out)


