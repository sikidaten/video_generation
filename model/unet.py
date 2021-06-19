import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32, activation=None, snnorm=False):
        super(UNet, self).__init__()

        features = init_features
        self.activation = activation if activation else nn.LeakyReLU(0.2,inplace=True)
        self.snnorm = snnorm
        self.encoder1 = self.block(in_channels, features)
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool1 = nn.Conv2d(features, features, kernel_size=1, stride=2)
        self.encoder2 = self.block(features, features * 2)
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.Conv2d(features * 2, features * 2, kernel_size=1, stride=2)
        self.encoder3 = self.block(features * 2, features * 4)
        # self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.Conv2d(features * 4, features * 4, kernel_size=1, stride=2)
        self.encoder4 = self.block(features * 4, features * 8)
        # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.Conv2d(features * 8, features * 8, kernel_size=1, stride=2)

        self.bottleneck = self.block(features * 8, features * 16)

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = self.block((features * 8) * 2, features * 8)
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = self.block((features * 4) * 2, features * 4)
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = self.block((features * 2) * 2, features * 2)
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = self.block(features * 2, features)

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
        self.linear = nn.Sequential(self.activation, nn.Linear(features * 16, 1))

    def forward(self, x):
        # x=x.permute(0,3,1,2)
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        x = self.conv(dec1)
        z = bottleneck.sum([2, 3])
        z=self.linear(z)
        return x,z

    def block(self, in_channels, features):
        conv1 = nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=3, padding=1, bias=False)
        conv2 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False, )
        # if self.snnorm:
        #     conv1, conv2 = spectral_norm(conv1), spectral_norm(conv2)
        return nn.Sequential(conv1, nn.BatchNorm2d(features), self.activation, conv2, nn.BatchNorm2d(features), self.activation)


if __name__ == "__main__":
    add_sn=lambda m:spectral_norm(m) if isinstance(m,(nn.Conv2d,nn.ConvTranspose2d)) else m
    model = UNet()
    model.apply(add_sn)
    print(model)
    output = model(torch.randn(1, 3, 64, 64))
    print(output[0].shape,output[1].shape)
