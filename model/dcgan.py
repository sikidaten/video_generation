import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, in_ch, feature, out_ch,acivation=nn.ReLU()):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(in_ch, feature * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature * 8),
            acivation,
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(feature * 8, feature * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature * 4),
            acivation,
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(feature * 4, feature * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature * 2),
            acivation,
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(feature * 2, feature, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature),
            acivation,
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(feature, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, x):
        x = self.main(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, in_ch, features,activaiton=nn.LeakyReLU(0.2,inplace=True)):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(in_ch, features, 4, 2, 1, bias=False),
            activaiton,
            # state size. (ndf) x 32 x 32
            nn.Conv2d(features, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            activaiton,
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(features * 2, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            activaiton,
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(features * 4, features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 8),
            activaiton,
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(features * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x):
        x = self.main(x)
        return x


from zviz import Zviz


class DCGAN(nn.Module):
    def __init__(self, optimizerG, optimizerD, lossDreal, lossDfake, lossG, zsize, feature,g_activation=nn.ReLU(inplace=True),d_activation=nn.LeakyReLU(0.2,inplace=True),enable_zviz=True):
        super(DCGAN, self).__init__()
        self.generator = Generator(zsize, feature, 3,acivation=g_activation)
        self.discriminator = Discriminator(3, feature,activaiton=d_activation)
        self.generator.apply(self.weights_init)
        self.discriminator.apply(self.weights_init)
        self.zviz = Zviz({'G': self.generator, 'D': self.discriminator} if enable_zviz else {})
        self.optG = optimizerG(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optD = optimizerD(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.zviz.setoptimizer(self.optG, 'optG')
        self.zviz.setoptimizer(self.optD, 'optD')
        self.lossDreal = lossDreal
        self.lossDfake = lossDfake
        self.lossG = lossG
        if not enable_zviz:self.zviz.disable_forever()

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
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
        self.zviz.step('optG')
        self.zviz.zero_grad('optG')
        self.zviz.zero_grad('optD')
        self.zviz.clear()
        self.zviz.disable_forever()
        return lossDreal.item(), lossDfake.item(), lossG.item(), fake.detach().cpu()


if __name__ == '__main__':
    model = Discriminator(3, 4)
    print(model(torch.randn(1, 3, 128, 128)).shape)
