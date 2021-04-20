from utils.util import MeanCoVariance_iter as MCVI
import numpy as np
import torch
mvi=MCVI()
b=8
B=8
C=2048
data=torch.randn(b,B,C,1,1)
# data=torch.arange(2*3*4).float().view(2,3,4,1,1)
print(data.view(b*B,C))
_data=data.reshape(b*B,C)
gtcov=np.cov(_data.numpy().transpose(),bias=True)
gtmean=_data.mean(dim=0)
print('gt',gtcov,gtmean)
# print('GT',_data.numpy().cov(unbiased=True),_data.mean(dim=0))
for i,d in enumerate(data):
    mvi.iter(d)
print(mvi.get())
print(mvi.get(False))