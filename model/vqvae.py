import torch
import torch.nn as nn
import torch.nn.functional as F


class VQDicFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *input):
        with torch.no_grad():
            z, z_dic = input

            def getnearrestdicidx(z, z_dic):
                B, F, H, W = z.shape
                _F, K = z_dic.shape
                assert F == _F
                ret = ((z.reshape(B, F, 1, H, W) - z_dic.reshape(1, F, K, 1, 1)) ** 2).mean(dim=1).argmin(dim=1)
                return ret

            idx = getnearrestdicidx(z, z_dic)
            return z_dic[:, idx].permute(1, 0, 2, 3), idx

    @staticmethod
    def backward(ctx, *grad_x):
        grad_x = grad_x[0]
        return grad_x, None


class VQDic(nn.Module):
    def __init__(self, z_dic):
        super(VQDic, self).__init__()
        self.z_dic = nn.Parameter(z_dic,requires_grad=False)

    def forward(self, *x):
        return VQDicFunction.apply(*x, self.z_dic)


class VQVAE(nn.Module):
    def __init__(self, optimizer, activation, dicsize, ):
        super(VQVAE, self).__init__()
        self.dicsize = dicsize
        modules = []
        hidden_dims = [32, 64, 128, 256, 512]
        z_feature = hidden_dims[-1]
        in_ch = 3
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    activation)
            )
            in_ch = h_dim
        modules.append(nn.Conv2d(z_feature,z_feature,1))
        self.encoder = nn.Sequential(*modules)
        modules = []
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(
            *modules,
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3,
                      kernel_size=3, padding=1),
            nn.Tanh())

        self.reconloss = nn.MSELoss()  # TODO really???
        self.optimizer = optimizer(self.parameters())
        self.z_dic = torch.randn(z_feature, self.dicsize)
        self.vqdic = VQDic(self.z_dic)

    def forward(self, x):
        z = self.encoder(x)
        self.z_shape = z.shape
        vqz, vqzidx = self.vqdic(z)
        x = self.decoder(vqz)
        return x, z, vqz, vqzidx

    def update_dic(self, z, vqzidx):
        for i in range(self.dicsize):
            if (vqzidx==i).any():
                self.z_dic[:, i] = z.permute(0, 2, 3, 1)[vqzidx == i].mean(0)

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

    def generate(self,randn):
        B,H,W=randn.shape
        # randn = torch.randint(0, self.dicsize, [B * H * W])
        vqz = self.z_dic[:, randn.reshape(-1)].reshape(-1, B, H, W).permute(1, 0, 2, 3).to(randn.device)
        return self.decoder(vqz)


if __name__ == '__main__':
    model = VQVAE(optimizer=torch.optim.Adam, activation=nn.ReLU(), dicsize=128)
    print(model)
    img = torch.randn(8, 3, 128, 128)
    loss = model.batch(img, 'train')
    print(loss)

    model.generate()

    # B, F, H, W = 8, 4, 16, 16
    # K = 32
    # z = torch.randn(B, F, H, W, requires_grad=True)
    # z_dic = torch.arange(F * K).reshape(F, K)
    # z_dic = torch.randn(F, K)
    # vqz, dicidx = VQDic(z_dic)(z)
    # print(f'{vqz.shape=}')
    # print(f'{dicidx.shape=}')
    # vqz.mean().backward()
    # print(z.grad.shape)
