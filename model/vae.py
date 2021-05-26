import torch
import torch.nn as nn
import torch.nn.functional as F


class VariationalAutoEncoder(nn.Module):
    def __init__(self, optimizer, activation, reconloss='bernolli', ):
        super(VariationalAutoEncoder, self).__init__()
        modules = []
        hidden_dims = [32, 64, 128, 256, 512]
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

        self.encoder = nn.Sequential(*modules)
        modules = []
        hidden_dims[-1]=hidden_dims[-1]//2
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
            nn.Tanh() if reconloss!='Bernoulli' else nn.Sigmoid())

        def lpxz_bernolli(data, target):
            with torch.no_grad():
                assert ((0 <= data)  & (data<= 1)).all(),data
                assert ((0 <= target) & (target <= 1)).all(),target
            return F.binary_cross_entropy(data, target)

        def lpxz_gaussian(data, target):
            pass
            # TODO

        self.reconloss = lpxz_bernolli if reconloss == 'Bernoulli' else lpxz_gaussian
        self.optimizer = optimizer(self.parameters())

    def parameterize(self, x):
        logvar, mu = x.chunk(2, dim=1)
        rand = torch.randn_like(logvar)
        return rand * logvar.exp() + mu, logvar, mu

    def forward(self, x):
        x = self.encoder(x)
        x, logvar, mu = self.parameterize(x)
        x = self.decoder(x)
        return x, logvar, mu

    def batch(self, img, phase):
        with torch.set_grad_enabled(phase == 'train'):
            recon, logvar, mu = self.forward(img)
            reconloss = self.reconloss(recon, img)
            KLDloss = -0.5 * (1 + 2*logvar  - mu ** 2 - logvar.exp()**2).mean()
            loss = reconloss + KLDloss
            if phase == 'train':
                loss.backward()
                self.optimizer.step()
                self.zero_grad()
        return {'loss': {f'recon_{phase}': reconloss.item(), f'KLD_{phase}': KLDloss.item()},
                'images': recon}

    def generate(self, x):
        return self.decoder(x.chunk(2,dim=1)[0])


if __name__ == '__main__':
    model = VariationalAutoEncoder(torch.optim.Adam, F.mse_loss, 128, nn.ReLU())
    print(model)
    img = torch.randn(4, 3, 128, 128)
    recon = model(img)
    print(recon.shape)
