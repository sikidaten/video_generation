import torch
import torch.nn as nn

from model.utils import ScaleConvBnRelu, GlobalAveragePooling


class Generator(nn.Module):
    def __init__(self, in_ch, feature, out_ch):
        super(Generator, self).__init__()
        # self.inconv = nn.Conv2d(in_ch, feature, 1)
        # self.upconv = nn.Sequential(*[ScaleConvBnRelu(feature, feature, 2) for _ in range(7)])
        # self.outconv = nn.Conv2d(feature, out_ch, 1)
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(in_ch, feature * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(feature * 8, feature * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(feature * 4, feature * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(feature * 2, feature, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(feature, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, x):
        # x = self.inconv(x)
        # x = self.upconv(x)
        # x = self.outconv(x)
        # x = torch.tanh(x)
        x=self.main(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, in_ch, features):
        super(Discriminator, self).__init__()
        # self.inconv = nn.Conv2d(in_ch, features, 1)
        # self.conv = nn.Sequential(*[nn.Sequential(*[ScaleConvBnRelu(features, features, 0.5),ScaleConvBnRelu(features,features,1)])for _ in range(3)])
        # self.gap = GlobalAveragePooling()
        # self.outconv = nn.Conv2d(features, 3, 1)
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(in_ch, features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(features, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(features * 2, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(features * 4, features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(features * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x):
        # x = self.inconv(x)
        # x = self.conv(x)
        # x = self.gap(x)
        # x = self.outconv(x)
        x=self.main(x)
        return x


class DCGAN(nn.Module):
    def __init__(self, optimizerG, optimizerD, lossDreal, lossDfake, lossG, zsize,feature):
        super(DCGAN, self).__init__()
        self.generator = Generator(zsize, feature, 3)
        self.discriminator = Discriminator(3, feature)
        self.optG = optimizerG(self.generator.parameters())
        self.optD = optimizerD(self.discriminator.parameters())
        self.lossDreal = lossDreal
        self.lossDfake = lossDfake
        self.lossG = lossG

    def trainbatch(self, noise, realimg, trainD=True):


        fake = self.generator(noise)
        if trainD:
            realout = self.discriminator(realimg)
            lossDreal = self.lossDreal(realout).mean()
            fakeout = self.discriminator(fake.detach())
            lossDfake = self.lossDfake(fakeout).mean()
            (lossDreal + lossDfake).backward()
            self.optD.step()
            self.optD.zero_grad()
        else:
            lossDfake = torch.tensor([0.])
            lossDreal = torch.tensor([0.])

        fakeout = self.discriminator(fake)
        lossG = self.lossG(fakeout).mean()
        lossG.backward()
        self.optG.step()
        self.optG.zero_grad()

        return lossDreal.item(), lossDfake.item(), lossG.item(), fake.detach().cpu()


if __name__ == '__main__':
    model = Discriminator(3, 4)
    print(model(torch.randn(1, 3, 128, 128)).shape)
