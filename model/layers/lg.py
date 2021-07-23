import torch
import torch.nn as nn
import torch.nn.functional as F

class lg_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, _min=-1, _max=1):
        # ctx._min = _min
        # ctx._max = _max
        # ctx.save_for_backward(x)
        return x.clamp(min=_min, max=_max)
        # return F.tanh(x)

    @staticmethod
    def backward(ctx, grad):
        # x = ctx.saved_tensors
        # x=torch.where((ctx._min<x)& (x<ctx._max),torch.ones(1).float().to(x.device),x)
        return grad


class LG(nn.Module):
    def __init__(self):
        super(LG, self).__init__()

    def forward(self, x):
        return lg_func.apply(x)


if __name__ == '__main__':
    x = torch.randn(8, requires_grad=True)
    m = LG()(x)
    print("output", m)
    m.sum().backward()
    print(x.grad)
