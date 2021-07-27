import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg

def min(x, a):
    return torch.stack([torch.zeros_like(x) + a, x]).min(0)[0]

def linerinterpolateroundlog2(mi,ma,n):
    ret=2**(np.round(np.log2((ma-mi)/(n-1)*np.arange(n)+mi))).astype(int)
    return np.where(ret<mi,mi,ret)
def max(x, a):
    return torch.stack([torch.zeros_like(x) + a, x]).max(0)[0]
import cloudpickle
def savecloudpickle(obj,path):
    with open(path,'wb') as f:
        f.write(cloudpickle.dumps(obj))
def loadcloudpickle(path):
    with open(path,'rb') as f:
        _obj=cloudpickle.loads(f.read())
    return _obj
# class MeanVariance_iter:
#     def __init__(self):
#         self.n = 0
#         self.s = 0
#         self.mu = 0
#
#     def iter(self, xn):
#         B, C, _, _ = xn.shape
#         xn = xn.view(B, C)
#         mun = (self.n * self.mu + xn.sum(dim=0)) / (self.n + B)
#         sn = (self.n * (self.s + self.mu ** 2) + (xn ** 2).sum(dim=0)) / (self.n + B) - mun ** 2
#         self.n += B
#         self.s = sn
#         self.mu = mun
#         return sn, mun

    # def get(self, isbias=True):
    #     bias = self.n / (self.n - 1) if isbias else 1
    #     return self.s * bias, self.mu * bias


class MeanCoVariance_iter:
    def __init__(self, device):
        self.n = 0
        self.s = 0
        self.mu = torch.zeros(1).to(device)

    def iter(self, xn):
        xn=xn.double()
        B, C, _, _ = xn.shape
        xn = xn.view(B, C)
        mun = (self.n * self.mu + xn.sum(dim=0)) / (self.n + B)
        sn = 1 / (self.n + B) * torch.einsum('ij,ik->jk', xn, xn) \
             + self.n / (self.n + B) * torch.einsum('i,j->ij', self.mu, self.mu) \
             - torch.einsum('i,j->ij', mun, mun) \
             + self.n / (self.n + B) * self.s
        self.s = sn
        self.n += B
        self.mu = mun

    def get(self, isbias=True):
        bias = self.n / (self.n - 1) if isbias else 1
        return self.s * bias, self.mu


def sqrtm(x):
    m = x.detach().cpu().numpy()
    sqrtm, _ = linalg.sqrtm(m, disp=False)
    return torch.from_numpy(sqrtm.real).to(x.device)


def make_gt_inception(model, loader, device):
    print('make inception output...')
    ret = []
    model = model.to(device)
    MCVI = MeanCoVariance_iter(device)
    for i, data in enumerate(loader):
        with torch.set_grad_enabled(False):
            print(f'\r{i},{len(loader)},{i/len(loader)*100:2.0f}%',end='')
            img = data[1]
            img = img.to(device)
            # print(img.shape)
            output = model(img)[0]
            MCVI.iter(output)
            ret.append(output)

    # ret = torch.cat(ret, dim=0)
    # ret = ret.reshape(-1, 2048)
    # scov, sm = torch.from_numpy(np.cov(ret.cpu(), rowvar=False)), ret.mean(dim=0)
    # mcov, mm = MCVI.get(isbias=False)
    # bcov, bmm = MCVI.get(isbias=True)
    # return scov,sm
    return MCVI.get(isbias=True)


def fid(cogtsigma, gtmean, cofakesigma, fakemean):
    return (torch.norm(gtmean - fakemean) ** 2 \
           + torch.trace(cogtsigma) + torch.trace(cofakesigma) - 2 * (torch.trace(sqrtm(cogtsigma @ cofakesigma)))).item()
def rgb_distance(x):
    return F.l1_loss(x.permute(0,3,2,1).reshape(-1,3).mean(0),torch.tensor([0.555,0.431,0.352])).item()
def similarity(x):
    B,C,H,W=x.shape
    x0=x.reshape(B,1,C,H,W)
    x1=x.reshape(1,B,C,H,W)
    return F.l1_loss(x0,x1)
def get_singular(module):
    ret={}
    for n,p in module.named_parameters():
        if 'weight' in n:
            _p=p
            ret[n]=torch.svd(_p.reshape(_p.shape[0],-1))[1].max().item()
    return ret
def get_singular_with_SN(model):
    ret={}
    params=model.state_dict()
    for key in params:
        if 'weight_orig' in key:
            _w=params[key]
            _u=params[key.replace('orig','u')]
            _v=params[key.replace('orig','v')]
            if _w.shape[1]==_u.shape[0]:
                _w=_w.permute(1,0,2,3)
            _w = _w.reshape(_w.shape[0], -1)
            ret[key]=_u@_w@_v
    assert ret!={}
    return ret
def normalize_grad(model):
    with torch.no_grad():
        for p in model.parameters():
            s=p.grad.shape
            p.grad=F.softmax(p.grad.view(-1)).view(s)

if __name__ == '__main__':
    print(linerinterpolateroundlog2(64,512,3))