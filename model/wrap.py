import torch.nn as nn

class Wrap(nn.Module):
    def __init__(self,model):
        super(Wrap, self).__init__()
        self.model=model
    def forward(self,x):
        x=self.model(x)
        return x.mean()