import torch
import torch.nn.functional as F
import numpy as np
import scipy
def min(x,a):
    return torch.stack([torch.zeros_like(x)+a,x]).min(0)[0]

def max(x,a):
    return torch.stack([torch.zeros_like(x)+a,x]).max(0)[0]

#TODO D
class MeanVariance_iter:
    def __init__(self):
        self.n=0
    def iter(self,s,mu,xn):
        B,C,_,_=xn.shape
        xn=xn.view(B,C)
        mun=(self.n*mu+xn.sum(dim=0))/(self.n+B)
        sn=(self.n*(s+mu**2)+(xn**2).sum(dim=0))/(self.n+B)-mun**2
        self.n+=xn.shape[0]
        return sn,mu
def covariancematrix(x):
    assert x.shape==1
    return x.T@x
def sqrtm(x):
    m = input.detach().cpu().numpy().astype(np.float_)
    sqrtm = torch.from_numpy(scipy.linalg.sqrtm(m).real).to(input)
    return sqrtm
#TODO D
def make_gt_inception(model,loader,resize=True,device='cpu'):
    model=model.to(device)
    mean=torch.zeros(2048)
    sigma=torch.zeros(2048)
    MVI=MeanVariance_iter()
    for i,data in enumerate(loader):
        print(i,len(loader))
        img=data[1]
        print(img.shape)
        if resize:
            img=F.interpolate(img,size=(299,299),mode='bilinear',align_corners=False)
        output=model(img.to(device))
        mean,sigma=MVI.iter(sigma,mean,output)
    return mean,sigma
def fid(gtsigma,gtmean,fakesigma,fakemean):
    cogtsigma=covariancematrix(gtsigma)
    cofakesigma=covariancematrix(fakesigma)
    return ((gtmean-fakemean)**2).mean()+torch.trace(cogtsigma+cofakesigma-2*sqrtm(cogtsigma@cofakesigma))

if __name__=='__main__':
    print(max(torch.tensor([-1,0,1,2]),0))