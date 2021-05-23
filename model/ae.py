import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    def __init__(self,optimizer,reconloss,activation):
        super(AutoEncoder, self).__init__()
        modules = []
        hidden_dims = [32, 64, 128, 256, 512]
        in_ch=3
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_channels=h_dim,kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    activation)
            )
            in_ch = h_dim

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
        self.reconloss=reconloss
        self.optimizer=optimizer(self.parameters())
    def forward(self,x):
        x=self.encoder(x)
        x=self.decoder(x)
        return x
    def batch(self,img,phase):
        with torch.set_grad_enabled(phase=='train'):
            recon=self.forward(img)
            reconloss=self.reconloss(recon,img)
            loss=reconloss
            if phase=='train':
                loss.backward()
                self.optimizer.step()
                self.zero_grad()
        return {'loss':{'recon':reconloss.item()},'images':recon*0.5+0.5}
    def generate(self,x):
        return self.decoder(x)*0.5+0.5

if __name__=='__main__':
    model=AutoEncoder(torch.optim.Adam,F.mse_loss,128,nn.ReLU())
    print(model)
    img=torch.randn(4,3,128,128)
    recon=model(img)
    print(recon.shape)