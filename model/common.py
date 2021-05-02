import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class InterpolateConv(nn.Module):
    def __init__(self, in_ch, out_ch, scale_factor, activate=nn.ReLU(inplace=True), batchnorm=True, snnorm=True):
        super(InterpolateConv, self).__init__()
        conv = nn.Conv2d(in_ch, out_ch, 3, bias=(batchnorm != None), padding=1)
        if snnorm: conv = spectral_norm(conv)
        self.main = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor),
            conv,
            nn.BatchNorm2d(out_ch) if batchnorm else nn.Identity(),
            activate
        )

    def forward(self, x):
        return self.main(x)


if __name__ == '__main__':
    model = InterpolateConv(128, 0.5)
    data = torch.randn(3, 128, 64, 64)
    output = model(data)
    print(output.shape)
