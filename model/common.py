import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


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
        elif scale_factor==2:
            # conv=nn.ConvTranspose2d(in_ch,out_ch,2,bias=not batchnorm,stride=2)
            conv=nn.Sequential(nn.Upsample(scale_factor=2),nn.Conv2d(in_ch,out_ch,3,bias=not batchnorm,padding=1))

        # if snnorm: conv = spectral_norm(conv)
        layers=[conv]
        if batchnorm:layers.append(nn.BatchNorm2d(out_ch))
        layers.append(activate)
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

if __name__ == '__main__':
    model = InterpolateConvcnn(128,128, 2)
    data = torch.randn(3, 128, 64, 64)
    output = model(data)
    print(output.shape)
