from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers.lg import LG
from model.layers.upsample_ident import UpSample_Ident
from utils.spectral_norm import spectral_norm
from core import Plotter
import utils.util as U

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
        self.upsample = UpSample_Ident(scale_factor)

    def forward(self, x):
        x = self.inconv(x)
        for layer in self.convs:
            x = layer(x)
        x = self.outconv(x)
        x = self.lastactivation(x)
        return x


class Generator(nn.Module):
    def __init__(self, in_ch, feature=64, activation=nn.ReLU(True)):
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
    def __init__(self, in_ch, feature=64, activation=nn.LeakyReLU(0.2, inplace=True), snnorm=False):
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
def savegrad(self, gradinput, gradoutput, modulename, baseself,size=64,thres=0.01):
    baseself.i_grad += 1
    with torch.no_grad():
        gout = gradoutput[0]
        if gout.max().abs() > thres: baseself.graddic[f'{modulename}:{baseself.i_grad}:max' + self.__class__.__name__] = gout.max().item()
        if gout.mean().abs() > thres: baseself.graddic[
            f'{modulename}:{baseself.i_grad}:mean' + self.__class__.__name__] = gout.mean().item()
        if gout.min().abs() > thres: baseself.graddic[f'{modulename}:{baseself.i_grad}:min' + self.__class__.__name__] = gout.min().item()
        # print(f'{modulename}:{baseself.i_grad}:{self.__class__.__name__}')
        gout = (gout - gout.min()) / (gout.max() - gout.min())
        gout = gout.abs().max(dim=0)[0].max(dim=0)[0].unsqueeze(0).unsqueeze(0)
        img = F.interpolate(gout, size=(size, size)).squeeze(0)
        baseself.gradimgs.append(img)




class DCGAN(nn.Module):
    def __init__(self, optimizerG, optimizerD, lossDreal, lossDfake, lossG, zsize, feature, size,
                 g_activation=nn.ReLU(inplace=True), d_activation=nn.LeakyReLU(0.2, inplace=True), enable_zviz=True,
                 discriminator=None, mode_seek_lambda=1, plotter=None):
        super(DCGAN, self).__init__()
        self.mode_seek_lambda = mode_seek_lambda
        # self.generator = BaseModel(in_ch=zsize, out_ch=3, feature=feature, scale_factor=2, size=size,
        #                            lastactivation=LG(), activation=g_activation, is_G=True,
        #                            norm_layer=nn.InstanceNorm2d)
        # self.discriminator = BaseModel(in_ch=3, out_ch=1, feature=feature, size=size, scale_factor=0.5,
        #                                lastactivation=nn.Identity(), activation=d_activation,
        #                                is_G=False,
        #                                norm_layer=nn.InstanceNorm2d) if discriminator is None else discriminator

        self.generator = Generator(in_ch=zsize,feature=64)
        self.discriminator = Discriminator(in_ch=3,feature=64)

        self.generator.apply(self.weights_init)
        self.discriminator.apply(self.weights_init)
        self.zviz = Zviz({'G': self.generator, 'D': self.discriminator} if enable_zviz else {})
        # self.optG = optimizerG(self.generator.parameters(), lr=0.00005, betas=(0, 0.999))
        # self.optD = optimizerD(self.discriminator.parameters(), lr=0.0002, betas=(0, 0.999))

        self.optG = optimizerG(self.generator.parameters(),lr=2e-4,betas=(0.5,0.999))
        self.optD = optimizerD(self.discriminator.parameters(),lr=2e-4,betas=(0.5,0.999))

        self.zviz.setoptimizer(self.optG, 'optG')
        self.zviz.setoptimizer(self.optD, 'optD')
        self.lossDreal = lossDreal
        self.lossDfake = lossDfake
        self.lossG = lossG
        self.plotter =plotter
        self.i_grad=0
        self.graddic={}
        self.gradimgs=[]
        if not enable_zviz: self.zviz.disable_forever()

        for m_name, modules in zip(['G', 'D'], [self.generator.named_modules(), self.discriminator.named_modules()]):
            for name, module in modules:
                print(module.__class__.__name__)
                if module.__class__.__name__ in ['Tanh', 'ConvTranspose2d', 'ReLU', 'BatchNorm2d', 'Conv2d', 'LeakyReLU']:
                    module.register_backward_hook(partial(savegrad, modulename=m_name,baseself=self))

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv2D') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm2D') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def trainbatch(self, noise, realimg,idx):
        # self.zviz.clear()
        B, C, H, W = realimg.shape
        fake = self.generator(noise)
        realimg.requires_grad = True
        realout = self.discriminator(realimg)
        lossDreal = self.lossDreal(realout).mean()
        fakeout = self.discriminator(fake.detach())
        lossDfake = self.lossDfake(fakeout).mean()
        # gradients_penalty = torch.autograd.grad(outputs=lossDreal, inputs=realimg, retain_graph=True)[0]
        # gradients_penalty = ((gradients_penalty * torch.randn_like(gradients_penalty, requires_grad=True))**2).mean()
        # gradients_penalty.backward()
        gradients_penalty=torch.zeros(1)
        self.i_grad=0
        self.graddic={}
        self.zviz.backward(lossDreal)
        self.plotter.add_scalars("Dreal",self.graddic,idx)
        self.i_grad=0
        self.graddic={}
        self.zviz.backward(lossDfake)
        self.plotter.add_scalars("Dfake",self.graddic,idx)
        if idx%10==0:
            dic={}
            for name,p in self.discriminator.named_parameters():
                if p.grad.max().abs()>0.1:dic[f'{name}:max']=p.grad.max().item()
                if p.grad.mean().abs() > 0.1: dic[f'{name}:mean'] = p.grad.mean().item()
                if p.grad.min().abs() > 0.1: dic[f'{name}:min'] = p.grad.min().item()
            self.plotter.add_scalars('D_grad',dic,idx)

        self.zviz.step('optD')
        self.zviz.zero_grad('optD')

        fakeout = self.discriminator(fake)
        lossG = self.lossG(fakeout).mean()
        self.i_grad=0
        self.graddic={}
        self.zviz.backward(lossG)
        self.plotter.add_scalars('G',self.graddic,idx)
        if idx%10==0:
            dic={}
            for name,p in self.generator.named_parameters():
                if p.grad.max().abs()>0.1:dic[f'{name}:max']=p.grad.max().item()
                if p.grad.mean().abs() > 0.1: dic[f'{name}:mean'] = p.grad.mean().item()
                if p.grad.min().abs() > 0.1: dic[f'{name}:min'] = p.grad.min().item()
            self.plotter.add_scalars('G_grad',dic,idx)
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
    # generator = Generator(in_ch=128)
    # discriminator = Discriminator(in_ch=3)
    print(discriminator)
    print(generator)
    # output = generator(torch.randn(8, 128, 1, 1))
    # print(discriminator)
    # output = discriminator(torch.randn(8, 3, size, size))
    # print(output.shape)
    # for n,_ in generator.named_parameters():
    #     print(n)
