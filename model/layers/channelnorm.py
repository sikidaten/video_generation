import torch
import torch.nn as nn
import torch.nn.functional as F

class ChNorm2D(nn.Module):
    def __init__(self,H,W):
        super(ChNorm2D, self).__init__()
        self.bn=nn.BatchNorm2d(H*W)
    def foward(self,x):
        B,C,H,W=x.shape
        x=x.reshape(B,C,H*W).permute(0,2,1)
        x=self.bn(x)
        return x