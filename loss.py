import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter


class SSIMLoss(nn.Module):
    def __init__(self, kernelsize=11, sigma=1.5):
        super(SSIMLoss, self).__init__()
        self.kernelsize = kernelsize
        self.sigma = sigma
        n = np.zeros((kernelsize, kernelsize))
        n[kernelsize // 2, kernelsize // 2] = 1
        self.gaussian_kernal = torch.from_numpy(gaussian_filter(n, sigma=sigma).astype(np.float32)).expand(3, 1,
                                                                                                           kernelsize,
                                                                                                           kernelsize)

    def forward(self, x, y):
        comp = lambda _x: F.conv2d(_x, self.gaussian_kernal.to(x.device), padding=self.kernelsize // 2, groups=3)
        ux = comp(x)
        uy = comp(y)

        uxx = comp(x ** 2)
        uyy = comp(y ** 2)
        uxy = comp(x * y)

        vx = uxx - ux ** 2
        vy = uyy - uy ** 2
        vxy = uxy - ux * uy

        c1 = 1e-4
        c2 = 9e-4
        numerator = (2 * ux * uy + c1) * (2 * vxy + c2)
        denominator = (ux ** 2 + uy ** 2 + c1) * (vx + vy + c2)
        return 1 - (numerator / (denominator + 1e-12)).mean()


if __name__ == '__main__':
    criterion = SSIMLoss()
    x = torch.randn(8, 3, 14, 14, requires_grad=True)
    y = torch.randn(8, 3, 14, 14, requires_grad=True)
    loss = criterion(x, x + torch.randn_like(x) * 0.2)
    loss.backward()
