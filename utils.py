import torch
def min(x,a):
    return torch.stack([torch.zeros_like(x)+a,x]).min(0)[0]

def max(x,a):
    return torch.stack([torch.zeros_like(x)+a,x]).max(0)[0]
if __name__=='__main__':
    print(max(torch.tensor([-1,0,1,2]),0))