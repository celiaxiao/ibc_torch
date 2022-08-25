import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Yihe's implementation of pointnet layer.
Used for processing particles/pointcloud information before appending to agent info.
Ref: https://github.com/fxia22/pointnet.pytorch
'''

class pointNetLayer(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()

        # global feature mlp
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 512, 1)
        # self.bn1 = nn.BatchNorm1d(64)
        # self.bn2 = nn.BatchNorm1d(128)
        # self.bn3 = nn.BatchNorm1d(512)

        # project to desired output dim
        self.mlp_out = nn.Linear(512, out_dim)
        # zero_init_output
        nn.init.zeros_(self.mlp_out.bias)
        self.mlp_out.weight.data.copy_(0.01 * self.mlp_out.weight.data)

    def forward(self, x):
        # print(x.size())
        assert x.size()[-1] == 3 # make sure last dim = 3
        x = x.transpose(2, 1)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x) # raw global feature

        # print(x.shape)
        
        x = torch.max(x, 2, keepdim=True)[0] # [B, 1024, 1]
        # print(x.shape)
        x = x.view(-1, 512) # global feature, [B, 1024]
        # print(x.shape)
        x = self.mlp_out(x) # [B, 512]
        return x
    

if __name__ == "__main__":
    B = 256 # batch size
    N = 704 # num of points
    input = torch.rand((B, N, 3))
    print(input.shape)
    ptnet = pointNetLayer()
    print(ptnet)
    # out = torch.squeeze(ptnet(input))
    out = ptnet(input)
    print(out.shape) # [N, 256]
    print(out)


