import torch
import torch.nn as nn
import torch.nn.functional as F
class InterpolateConv(nn.Module):
    def __init__(self,feature,scale_factor,activate=nn.ReLU(inplace=True),normalize=None,in_ch=None,out_ch=None):
        super(InterpolateConv, self).__init__()
        if in_ch is None:in_ch=feature
        if out_ch is None:out_ch=feature
        self.conv=nn.Conv2d(in_ch,out_ch,3,bias=(normalize!=None),padding=1)
        self.activate=activate
        self.normalize=nn.BatchNorm2d(out_ch) if normalize is None else normalize
        self.scale_factor=scale_factor

    def forward(self,x):
        x=F.interpolate(x,scale_factor=self.scale_factor)
        x=self.conv(x)
        x=self.normalize(x)
        x=self.activate(x)
        return x

if __name__=='__main__':
    model=InterpolateConv(128,0.5)
    data=torch.randn(3,128,64,64)
    output=model(data)
    print(output.shape)