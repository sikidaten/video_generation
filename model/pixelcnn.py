import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedConv2D(nn.Module):
    def __init__(self, in_ch, out_ch, k):
        super(MaskedConv2D, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k)
        mask = torch.zeros(in_ch, k, k)
        mask[:, :k // 2, :] = 1
        mask[:, k // 2, :k // 2] = 1
        self.mask = nn.Parameter(mask)
        self.k = k

    def forward(self, x):
        x = F.pad(x, [(self.k - 1) // 2] * 4)
        return F.conv2d(x, self.conv.weight * self.mask, self.conv.bias)


class MaskedAtention(nn.Module):
    def __init__(self, feature, num_heads=1):
        super(MaskedAtention, self).__init__()
        self.attn = nn.TransformerEncoderLayer(feature, num_heads)
        print(self.attn)

    def forward(self, x):
        B, C, H, W = x.shape
        mask = torch.ones(H * W, H * W).triu().bool()
        mask[0, 0] = False
        return self.attn(src=x.reshape(B, C, -1).permute(2, 0, 1), src_mask=mask).permute(1, 2, 0).reshape(B, C, H, W)


class NaivePixelCNN(nn.Module):
    def __init__(self, feature, num_layer=5, activaton=nn.Hardswish(), optimizer=torch.optim.Adam, sm='-1_1',
                 lossf=None):
        super(NaivePixelCNN, self).__init__()
        self.lossf = lossf
        layer = []
        layer.append(MaskedConv2D(3, feature, num_layer))
        for i in range(5):
            layer.append(MaskedConv2D(feature, feature, 5))
            layer.append(nn.BatchNorm2d(feature))
            layer.append(activaton)
        layer.append(MaskedConv2D(feature, 256 * 3, 5))
        self.layer = nn.ModuleList(layer)
        self.optimizer = optimizer(self.layer.parameters())
        self.sm = sm
        self.s,self.m=(0.5,0.5) if sm=='-1_1' else (1,0)

    def forward(self, x):
        for l in self.layer:
            x = l(x)
        return x

    def loss(self, img):
        output = self.forward(img)
        B, C, H, W = output.shape
        if self.lossf is not None:
            return self.lossf(output, img)
        elif self.sm == '-1_1':
            return F.cross_entropy(output.reshape(B, 3, C // 3, H, W).permute(0, 2, 1, 3, 4),
                                   ((img * 0.5 + 0.5) * 255).long()), (
                               output.reshape(B, 3, C // 3, H, W).argmax(2) / 255 - 0.5) / 0.5

    def batch(self, img, phase):
        with torch.set_grad_enabled(phase == 'train'):
            loss, outimg = self.loss(img)
            if phase=='train':
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            return {'loss': {'recon': loss.item()}, 'images': outimg}

    def generate(self, size, device='cpu',num=1,B=1):
        self.eval()
        ret= torch.zeros(B, 3, size, size).to(device)
        with torch.no_grad():
            for i in range(size):
                for j in range(size):
                    output = self.forward(ret)
                    ret[:,:,i,j]=(F.softmax(output[:, :, i, j].reshape(B * 3, 256), dim=-1).multinomial(1).reshape(B, 3)-self.m)/self.s
        self.train()
        return ret


if __name__ == '__main__':
    model = MaskedConv2D(1, 2, 5)
    model = NaivePixelCNN(128)
    print(model.generate(7).shape)
    optimizer = torch.optim.Adam(model.parameters())
    # B, C, H, W = 1, 3, 7, 7
    # data = (torch.rand((B,C,H,W))-0.5)/0.5
    # loss = model.loss(data)
    # (loss).mean().backward()
    # optimizer.step()
