import numpy as np
import torch
import torch.nn as nn

import utils.util as U
from model.common import InterpolateConv


class BaseModel(nn.Module):
    def __init__(self, in_ch, out_ch, feature, size, scale_factor, lastactivation, activation, is_G=False):
        super(BaseModel, self).__init__()
        numrange = int(np.log2(size))
        features = U.linerinterpolateroundlog2(feature, 512, numrange+1)
        if is_G: features = features[::-1]
        self.inconv = nn.Conv2d(in_ch, features[0], 1)
        self.convs = nn.Sequential(
            *[InterpolateConv(in_ch=features[i], out_ch=features[i + 1], scale_factor=scale_factor, activate=activation,snnorm=True)
              for i in range(numrange)])
        self.outconv = nn.Conv2d(features[-1], out_ch, 3, padding=1)
        self.lastactivation = lastactivation

    def forward(self, x):
        x = self.inconv(x)
        x = self.convs(x)
        x = self.outconv(x)
        x = self.lastactivation(x)
        return x


class Generator(nn.Module):
    def __init__(self, in_ch, feature, size, out_ch=3, activation=nn.ReLU(), lastactivation=nn.Tanh()):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_ch, feature * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature * 8),
            activation,
            nn.ConvTranspose2d(feature * 8, feature * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature * 4),
            activation,
            nn.ConvTranspose2d(feature * 4, feature * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature * 2),
            activation,
            nn.ConvTranspose2d(feature * 2, feature, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature),
            activation,
            nn.ConvTranspose2d(feature, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, in_ch, feature, size, out_ch=1, activation=nn.LeakyReLU(0.2, inplace=True)):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_ch, feature, 4, 2, 1, bias=False),
            activation,
            nn.Conv2d(feature, feature * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature * 2),
            activation,
            nn.Conv2d(feature * 2, feature * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature * 4),
            activation,
            nn.Conv2d(feature * 4, feature * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature * 8),
            activation,
            nn.Conv2d(feature * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x):
        return self.main(x)


from zviz import Zviz


class DCGAN(nn.Module):
    def __init__(self, optimizerG, optimizerD, lossDreal, lossDfake, lossG, zsize, feature, size,
                 g_activation=nn.ReLU(inplace=True), d_activation=nn.LeakyReLU(0.2, inplace=True), enable_zviz=True,
                 discriminator=None):
        super(DCGAN, self).__init__()
        self.generator = Generator(zsize, feature, 3, activation=g_activation)
        self.discriminator = discriminator if discriminator else Discriminator(3, feature, activation=d_activation,size=size)
        # self.generator = BaseModel(in_ch=zsize, out_ch=3, feature=feature, scale_factor=2, size=size,
        #                            lastactivation=nn.Tanh(), activation=g_activation,is_G=True)
        self.discriminator = BaseModel(in_ch=3, out_ch=1, feature=feature, size=size, scale_factor=0.5,
                                       lastactivation=nn.Identity(), activation=d_activation,
                                       is_G=False) if discriminator is None else discriminator
        # self.generator.apply(self.weights_init)
        # self.discriminator.apply(self.weights_init)
        self.zviz = Zviz({'G': self.generator, 'D': self.discriminator} if enable_zviz else {})
        self.optG = optimizerG(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optD = optimizerD(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.zviz.setoptimizer(self.optG, 'optG')
        self.zviz.setoptimizer(self.optD, 'optD')
        self.lossDreal = lossDreal
        self.lossDfake = lossDfake
        self.lossG = lossG
        if not enable_zviz: self.zviz.disable_forever()

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv2D') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm2D') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def trainbatch(self, noise, realimg, trainD=True):
        # self.zviz.clear()
        fake = self.generator(noise)
        if trainD:
            realout = self.discriminator(realimg)
            lossDreal = self.lossDreal(realout).mean()
            fakeout = self.discriminator(fake.detach())
            lossDfake = self.lossDfake(fakeout).mean()
            self.zviz.backward(lossDfake)
            self.zviz.backward(lossDreal)
            self.zviz.step('optD')
            self.zviz.zero_grad('optD')
        else:
            lossDfake = torch.tensor([0.])
            lossDreal = torch.tensor([0.])

        fakeout = self.discriminator(fake)
        lossG = self.lossG(fakeout).mean()
        self.zviz.backward(lossG)
        if lossDfake<1e-3:self.zviz.step('optG')
        self.zviz.zero_grad('optG')
        self.zviz.zero_grad('optD')
        self.zviz.clear()
        self.zviz.disable_forever()
        return lossDreal.item(), lossDfake.item(), lossG.item(), fake.detach().cpu()


if __name__ == '__main__':
    size = 128
    generator = BaseModel(in_ch=128, out_ch=3, feature=128, scale_factor=2, size=size, lastactivation=nn.Tanh(),
                          activation=nn.ReLU())
    discriminator = BaseModel(in_ch=3, out_ch=1, feature=128, size=size, scale_factor=0.5, lastactivation=nn.Identity(),
                              activation=nn.ReLU(),
                              is_G=False,)
    # print(generator)
    # output = generator(torch.randn(1, 128, 1, 1))
    print(discriminator)
    output = discriminator(torch.randn(8, 3, size, size))
    print(output.shape)
