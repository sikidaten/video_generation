import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.common import CNA
from model.layers.lg import LG
from utils.spectral_norm import spectral_norm


class BaseModel(nn.Module):
    def __init__(self, in_ch, out_ch, feature, size, scale_factor, lastactivation, activation, norm_layer, is_G=False,
                 snnorm=False):
        super(BaseModel, self).__init__()
        numrange = int(np.log2(size))
        self.inconv = nn.Conv2d(in_ch, feature, 1)
        self.convs = nn.ModuleList(
            [CNA(feature=feature, kernel=3, norm_layer=norm_layer,
                 activation=activation, scale_factor=scale_factor) for i in range(numrange)])
        self.outconv = nn.Conv2d(feature, out_ch, 3, padding=1)
        if snnorm: self.inconv = spectral_norm(self.inconv)
        if snnorm: self.outconv = spectral_norm(self.outconv)
        self.lastactivation = lastactivation
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.inconv(x)
        for layer in self.convs:
            x = layer(x) + F.upsample_bilinear(x, scale_factor=self.scale_factor)
        x = self.outconv(x)
        x = self.lastactivation(x)
        return x


class Generator(nn.Module):
    def __init__(self, in_ch, feature, size, out_ch=3, activation=nn.ReLU(), lastactivation=nn.Tanh(), snnorm=False):
        super(Generator, self).__init__()
        conv0 = nn.ConvTranspose2d(in_ch, feature * 8, 4, 1, 0, bias=False)
        conv0 = spectral_norm(conv0)
        self.main = nn.Sequential(
            conv0,
            nn.BatchNorm2d(feature * 8),
            activation,
            spectral_norm(nn.ConvTranspose2d(feature * 8, feature * 4, 4, 2, 1, bias=False), enable=snnorm),
            nn.BatchNorm2d(feature * 4),
            activation,
            spectral_norm(nn.ConvTranspose2d(feature * 4, feature * 2, 4, 2, 1, bias=False), enable=snnorm),
            nn.BatchNorm2d(feature * 2),
            activation,
            spectral_norm(nn.ConvTranspose2d(feature * 2, feature, 4, 2, 1, bias=False), enable=snnorm),
            nn.BatchNorm2d(feature),
            activation,
            spectral_norm(nn.ConvTranspose2d(feature, 3, 4, 2, 1, bias=False), enable=snnorm),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, in_ch, feature, size, out_ch=1, activation=nn.LeakyReLU(0.2, inplace=True), snnorm=False):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            spectral_norm(nn.Conv2d(in_ch, feature, 4, 2, 1, bias=False), enable=snnorm),
            activation,
            spectral_norm(nn.Conv2d(feature, feature * 2, 4, 2, 1, bias=False), enable=snnorm),
            nn.BatchNorm2d(feature * 2),
            activation,
            spectral_norm(nn.Conv2d(feature * 2, feature * 4, 4, 2, 1, bias=False), enable=snnorm),
            nn.BatchNorm2d(feature * 4),
            activation,
            spectral_norm(nn.Conv2d(feature * 4, feature * 8, 4, 2, 1, bias=False), enable=snnorm),
            nn.BatchNorm2d(feature * 8),
            activation,
            spectral_norm(nn.Conv2d(feature * 8, 1, 4, 1, 0, bias=False), enable=snnorm),
        )

    def forward(self, x):
        return self.main(x)


from zviz import Zviz


class DCGAN(nn.Module):
    def __init__(self, optimizerG, optimizerD, lossDreal, lossDfake, lossG, zsize, feature, size,
                 g_activation=nn.ReLU(inplace=True), d_activation=nn.LeakyReLU(0.2, inplace=True), enable_zviz=True,
                 discriminator=None, mode_seek_lambda=1):
        super(DCGAN, self).__init__()
        self.mode_seek_lambda = mode_seek_lambda
        # self.generator = Generator(zsize, feature, 3, activation=g_activation)
        # self.discriminator = discriminator if discriminator else Discriminator(3, feature, activation=d_activation,size=size)
        self.generator = BaseModel(in_ch=zsize, out_ch=3, feature=feature, scale_factor=2, size=size,
                                   lastactivation=LG(), activation=g_activation, is_G=True,norm_layer=nn.InstanceNorm2d)
        self.discriminator = BaseModel(in_ch=3, out_ch=1, feature=feature, size=size, scale_factor=0.5,
                                       lastactivation=nn.Identity(), activation=d_activation,
                                       is_G=False,norm_layer=nn.InstanceNorm2d) if discriminator is None else discriminator
        # self.generator.apply(self.weights_init)
        # self.discriminator.apply(self.weights_init)
        self.zviz = Zviz({'G': self.generator, 'D': self.discriminator} if enable_zviz else {})
        self.optG = optimizerG(self.generator.parameters(), lr=0.00005, betas=(0, 0.999))
        self.optD = optimizerD(self.discriminator.parameters(), lr=0.0002, betas=(0, 0.999))

        # print(self.generator)
        # exit()
        # self.optG = optimizerG(self.generator.parameters(),lr=1e-5)
        # self.optD = optimizerD(self.discriminator.parameters(),lr=1e-4)
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

    def trainbatch(self, noise, realimg):
        # self.zviz.clear()
        B, C, H, W = realimg.shape
        fake = self.generator(noise)
        realimg.requires_grad = True
        realout = self.discriminator(realimg)
        lossDreal = self.lossDreal(realout).mean()
        fakeout = self.discriminator(fake.detach())
        lossDfake = self.lossDfake(fakeout).mean()
        gradients_penalty = torch.autograd.grad(outputs=lossDreal, inputs=realimg, retain_graph=True)[0]
        gradients_penalty = (gradients_penalty * torch.randn_like(gradients_penalty, requires_grad=True)).mean()
        gradients_penalty.backward()
        self.zviz.backward(lossDreal)
        self.zviz.backward(lossDfake)
        self.zviz.step('optD')
        self.zviz.zero_grad('optD')

        fakeout = self.discriminator(fake)
        lossG = self.lossG(fakeout).mean()
        self.zviz.backward(lossG)
        self.zviz.step('optG')
        self.zviz.zero_grad('optG')
        self.zviz.zero_grad('optD')
        self.zviz.clear()
        self.zviz.disable_forever()
        return {'loss': {'Dreal': lossDreal.item(), 'Dfake': lossDfake.item(), 'G': lossG.item(),
                         "R1": gradients_penalty.item()}, 'image': {'fake': fake.detach().cpu()}}


if __name__ == '__main__':
    size = 256
    generator = BaseModel(in_ch=128, out_ch=3, feature=128, scale_factor=2, size=size, lastactivation=nn.Tanh(),
                          activation=nn.ReLU(), norm_layer=nn.BatchNorm2d)
    discriminator = BaseModel(in_ch=3, out_ch=1, feature=128, size=size, scale_factor=0.5, lastactivation=nn.Identity(),
                              activation=nn.ReLU(),
                              is_G=False, norm_layer=nn.BatchNorm2d)
    print(generator)
    output = generator(torch.randn(8, 128, 1, 1))
    # print(discriminator)
    # output = discriminator(torch.randn(8, 3, size, size))
    print(output.shape)
