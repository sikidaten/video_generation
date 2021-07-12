import torch
import torch.nn as nn
import torch.nn.functional as F


class NaiveMaskedConv2D(nn.Module):
    def __init__(self, in_ch, out_ch, k):
        super(NaiveMaskedConv2D, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k)
        mask = torch.zeros(in_ch, k, k)
        mask[:, :k // 2, :] = 1
        mask[:, k // 2, :k // 2] = 1
        self.mask = nn.Parameter(mask)
        self.k = k

    def forward(self, x):
        x = F.pad(x, [(self.k - 1) // 2] * 4)
        return F.conv2d(x, self.conv.weight * self.mask, self.conv.bias)


class GatedActivation(nn.Module):
    def __init__(self):
        super(GatedActivation, self).__init__()

    def forward(self, x):
        x0, x1 = x.chunk(2, dim=1)
        return F.tanh(x0) * F.sigmoid(x1)


class GatedConv2D(nn.Module):
    def __init__(self, feature, kernel, is_causal=False):
        super(GatedConv2D, self).__init__()
        self.kernel=kernel
        self.v_conv = nn.Conv2d(feature, feature* 2, (kernel // 2 + 1, kernel), padding=(kernel // 2, kernel // 2))
        self.h_conv = nn.Conv2d(feature, feature * 2, (1, kernel // 2 + 1), padding=(0, kernel // 2))
        self.h_conv2 = nn.Conv2d(feature, feature, (1, 1))
        self.v_h_conv = nn.Conv2d(feature * 2, feature * 2, (1, 1))
        self.activation = GatedActivation()

        if is_causal:
            self.h_conv.weight.data[:, :, -1, -1].zero_()

    def forward(self, x):
        vx,hx=x
        vx = self.v_conv(vx)[:,:,:-self.kernel//2+1]
        vout = self.activation(vx)
        _hx = self.h_conv(hx)[:,:,:,:-self.kernel//2+1] + vx
        _hx = self.activation(_hx)
        _hx = self.h_conv2(_hx)
        hout = hx + _hx
        return vout, hout


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


class PixelCNN(nn.Module):
    def __init__(self, feature, num_layer=20, activaton=nn.Hardswish(), optimizer=torch.optim.Adam, sm='-1_1',
                 lossf=None):
        super(PixelCNN, self).__init__()
        self.lossf = lossf
        layer = []
        layer.append(nn.Conv2d(3,feature,1))
        layer.append(GatedConv2D(feature, 5, is_causal=True))
        for i in range(num_layer):
            layer.append(GatedConv2D(feature, 5))
            # layer.append(nn.BatchNorm2d(feature))
            # layer.append(activaton)
        layer.append(nn.Conv2d(feature, 256 * 3, 1))
        self.layer = nn.ModuleList(layer)
        print(self.layer)
        self.optimizer = optimizer(self.layer.parameters())
        self.sm = sm
        self.s, self.m = (0.5, 0.5) if sm == '-1_1' else (1, 0)

    def forward(self, x):
        for idx,l in enumerate(self.layer):
            if idx==1:
                x=l([x,x])
            elif idx==len(self.layer)-1:
                x=l(x[1])
            else:
                x=l(x)
        return x

    def loss(self, img):
        output = self.forward(img)
        B, C, H, W = output.shape
        if self.lossf is not None:
            return self.lossf(output, img)
        elif self.sm == '-1_1':
            return F.cross_entropy(output.reshape(B, 3, 256, H, W).permute(0, 2, 1, 3, 4),
                                   ((img * 0.5 + 0.5) * 255).long()), (
                               output.reshape(B, 3, 256, H, W).argmax(2) / 255 - 0.5) / 0.5

    def batch(self, img, phase):
        with torch.set_grad_enabled(phase == 'train'):
            loss, outimg = self.loss(img)
            if phase == 'train':
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            return {'loss': {'recon': loss.item()}, 'images': outimg}

    def generate(self, size, device='cpu', B=1):
        self.eval()
        ret = torch.zeros(B, 3, size, size).to(device)
        with torch.no_grad():
            for i in range(size):
                for j in range(size):
                    output = self.forward(ret)
                    ret[:, :, i, j] = (F.softmax(output[:, :, i, j].reshape(B * 3, 256), dim=-1).multinomial(1).reshape(
                        B, 3) / 255 - self.m) / self.s
        self.train()
        return ret


if __name__ == '__main__':
    model = PixelCNN(128)
    optimizer = torch.optim.Adam(model.parameters())
    B, C, H, W = 1, 3, 128, 128
    data = (torch.rand((B, C, H, W)) - 0.5) / 0.5
    loss = model.loss(data)
    (loss).mean().backward()
    optimizer.step()
