import torch.nn as nn
import torch
import torch.nn.functional as F

class ScaleConvBnRelu(nn.Module):
    def __init__(self,inch,outch,scale_factor):
        super(ScaleConvBnRelu, self).__init__()
        self.conv=nn.Conv2d(inch,outch,3,1,1)
        self.bn=nn.BatchNorm2d(outch)
        self.scale_factor=scale_factor

    def forward(self,x):
        x=F.interpolate(x,scale_factor=self.scale_factor)
        x=self.conv(x)
        x=self.bn(x)
        x=F.relu(x)
        return x

class GlobalAveragePooling(nn.Module):
    def __init__(self):
        super(GlobalAveragePooling, self).__init__()

    def forward(self,x):
        x=F.interpolate(x,size=1)
        return x




if __name__=='__main__':
    model=GlobalAveragePooling()
    print(model(torch.randn(3,3,8,8)).shape)