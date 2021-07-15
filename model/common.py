import torch
import torch.nn as nn
from utils.spectral_norm import spectral_norm
from model.resnet import Bottleneck,BasicBlock
class _sqlinear(torch.autograd.Function):
    def __init__(self,cut=0,linear=1):
        self.cut=cut
        self.linear=linear
    @staticmethod
    def forward(self,x):
        self.save_for_backward(x.clone())
        x[x<0]=0
        x[(0<x) & (x<1)]=x[(0<x) & (x<1)]**2
        return x

    @staticmethod
    def backward(self,g):
        x=self.saved_tensors[0]
        x[x<0]=0
        x[x>1]=1
        x[(0<x) & (x<1)]*=2
        g=g.clone()
        return g*x
class SQLinear(nn.Module):
    def __init__(self,cut=0,linear=1):
        super(SQLinear, self).__init__()
        self.cut=cut
        self.linear=linear
    def forward(self,x):
        return _sqlinear(cut=self.cut,linear=self.linear).apply(x)

class BAC(nn.Module):
    def __init__(self,feature,kernel,activation):
        super(BAC, self).__init__()
        self.bn=nn.BatchNorm2d(feature)
        self.activation=activation
        self.conv=nn.Conv2d(feature,feature,kernel,padding=(kernel-1)//2)
    def forward(self,x):
        x=self.bn(x)
        x=self.activation(x)
        x=self.conv(x)
        return x
class InterpolateConv(nn.Module):
    def __init__(self, in_ch, out_ch, scale_factor, activate=nn.ReLU(inplace=True), batchnorm=True, snnorm=True):
        super(InterpolateConv, self).__init__()
        conv = nn.Conv2d(in_ch, out_ch, 3, bias=(batchnorm != None), padding=1)
        # if snnorm: conv = spectral_norm(conv)
        self.main = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor,mode='area'),
            conv,
            nn.BatchNorm2d(out_ch) if batchnorm else nn.Identity(),
            activate
        )

    def forward(self, x):
        return self.main(x)

class InterpolateConvcnn(nn.Module):
    def __init__(self, in_ch, out_ch, scale_factor, activate=nn.ReLU(inplace=True), batchnorm=True, snnorm=True):
        super(InterpolateConvcnn, self).__init__()
        if scale_factor==0.5:
            conv = nn.Conv2d(in_ch, out_ch, 3, bias=not batchnorm, padding=1,stride=2)
            if snnorm: conv = spectral_norm(conv)
        elif scale_factor==2:
            # conv=nn.ConvTranspose2d(in_ch,out_ch,2,bias=not batchnorm,stride=2)
            cnn=nn.Conv2d(in_ch,out_ch,3,bias=not batchnorm,padding=1)
            if snnorm: cnn = spectral_norm(cnn)
            conv=nn.Sequential(nn.Upsample(scale_factor=2),cnn)

        layers=[conv]
        if batchnorm:layers.append(nn.BatchNorm2d(out_ch))
        layers.append(activate)
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

if __name__ == '__main__':
    model=SQLinear()
    data = torch.randn(4)*2
    data.requires_grad=True
    output = model(data)
    output.sum().backward()
    print(data)
    print(output)
    print(data.grad)