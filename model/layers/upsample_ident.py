import torch
import torch.nn as nn
import torch.nn.functional as F

class upsample_ident_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,scale_factor):
        ctx.scale_factor=scale_factor
        return F.upsample(x,scale_factor=scale_factor)
    @staticmethod
    def backward(ctx,grad):
        ret=F.upsample(grad,scale_factor=1/ctx.scale_factor)
        return ret,None

class UpSample_Ident(nn.Module):
    def __init__(self,scale_factor):
        super(UpSample_Ident, self).__init__()
        self.scale_factor=scale_factor
    def forward(self,x):
        return upsample_ident_func.apply(x,self.scale_factor)

if __name__=="__main__":
    x=torch.ones(1,1,4,4,requires_grad=True)
    u=UpSample_Ident(2)
    u(x).sum().backward()
    print(x.grad)