import torch
import torch.nn as nn
import torch.nn.functional as F

from model.vqvae_vae import VQDic


class ResBlock(nn.Module):
    def __init__(self, feature, conv, activation):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            activation,
            conv(feature, feature, 3, 1, 1),
            nn.BatchNorm2d(feature),
            activation,
            conv(feature, feature, 1),
            nn.BatchNorm2d(feature),
        )

    def forward(self, x):
        return x + self.block(x)


class VQVAE(nn.Module):
    def __init__(self, feature, z_feature, optimizer, activation, zdicsize=512):
        super(VQVAE, self).__init__()
        self.zdicsize = zdicsize

        def makemodule(conv, in_ch, out_ch): return nn.Sequential(
            conv(in_ch, feature, 4, 2, 1),
            nn.BatchNorm2d(feature),
            activation,
            conv(feature, feature, 4, 2, 1),
            ResBlock(feature, nn.Conv2d, activation),
            ResBlock(feature, nn.Conv2d, activation),
            conv(feature, out_ch, 1)
        )

        self.encoder = makemodule(nn.Conv2d, 3, z_feature)
        self.decoder = nn.Sequential(makemodule(nn.ConvTranspose2d, z_feature, 3), nn.Tanh())

        self.reconloss = nn.MSELoss()  # TODO really???
        self.optimizer = optimizer(self.parameters())
        self.z_dic = torch.randn(z_feature, self.zdicsize)
        self.vqdic = VQDic(self.z_dic)

    def forward(self, x):
        z = self.encoder(x)
        vqz, vqzidx = self.vqdic(z)
        x = self.decoder(vqz)
        return x, z, vqz, vqzidx

    def batch(self, img, phase):
        with torch.set_grad_enabled(phase == 'train'):
            recon, z, vqz, vqzidx = self.forward(img)
            reconloss = self.reconloss(recon, img)
            KLDloss = F.mse_loss(z, vqz.detach())
            loss = reconloss + 0.25 * KLDloss
            if phase == 'train':
                loss.backward()
                self.optimizer.step()
                self.update_dic(z, vqzidx)
                self.zero_grad()
        return {'loss': {f'recon_{phase}': reconloss.item(), f'KLD_{phase}': KLDloss.item()},
                'images': recon}

    def generate(self, randn):
        B, H, W = randn.shape
        vqz = self.z_dic[:, randn.reshape(-1)].reshape(-1, B, H, W).permute(1, 0, 2, 3).to(randn.device)
        return self.decoder(vqz)

    def update_dic(self, z, vqzidx):
        for i in range(self.zdicsize):
            if (vqzidx == i).any():
                self.z_dic[:, i] = z.permute(0, 2, 3, 1)[vqzidx == i].mean(0)
