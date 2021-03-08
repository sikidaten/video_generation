import torch
import torch.nn as nn

from model.utils import ScaleConvBnRelu, GlobalAveragePooling


class Generator(nn.Module):
    def __init__(self, in_ch, feature, out_ch):
        super(Generator, self).__init__()
        self.inconv = nn.Conv2d(in_ch, feature, 1)
        self.upconv = nn.Sequential(*[ScaleConvBnRelu(feature, feature, 2) for _ in range(7)])
        self.outconv = nn.Conv2d(feature, out_ch, 1)

    def forward(self, x):
        x = self.inconv(x)
        x = self.upconv(x)
        x = self.outconv(x)
        x = torch.tanh(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, in_ch, features):
        super(Discriminator, self).__init__()
        self.inconv = nn.Conv2d(in_ch, features, 1)
        self.conv = nn.Sequential(*[nn.Sequential(*[ScaleConvBnRelu(features, features, 0.5),ScaleConvBnRelu(features,features,1)])for _ in range(3)])
        self.gap = GlobalAveragePooling()
        self.outconv = nn.Conv2d(features, 3, 1)

    def forward(self, x):
        x = self.inconv(x)
        x = self.conv(x)
        x = self.gap(x)
        x = self.outconv(x)
        return x


class DCGAN(nn.Module):
    def __init__(self, optimizerG, optimizerD, lossDreal, lossDfake, lossG, zsize,feature):
        super(DCGAN, self).__init__()
        self.generator = Generator(zsize, feature, 3)
        self.discriminator = Discriminator(3, feature)
        self.optG = optimizerG(self.parameters())
        self.optD = optimizerD(self.parameters())
        self.lossDreal = lossDreal
        self.lossDfake = lossDfake
        self.lossG = lossG

    def trainbatch(self, noise, realimg, trainD=True):
        lossDfake = torch.tensor([0.])
        lossDreal = torch.tensor([0.])
        fake = self.generator(noise)
        if trainD:
            # train D
            self.optD.zero_grad()
            realout = self.discriminator(realimg)
            lossDreal = self.lossDreal(realout).mean()
            lossDreal.backward()
            fakeout = self.discriminator(fake.detach())
            lossDfake = self.lossDfake(fakeout).mean()
            lossDfake.backward()
            # (lossDreal + lossDfake).backward()
            self.optD.step()

        # train G
        self.optG.zero_grad()
        fakeout = self.discriminator(fake)
        lossG = self.lossG(fakeout).mean()
        lossG.backward()
        self.optG.step()

        return lossDreal.item(), lossDfake.item(), lossG.item(), fake.detach().cpu()


if __name__ == '__main__':
    model = Discriminator(3, 4)
    print(model(torch.randn(1, 3, 128, 128)).shape)
