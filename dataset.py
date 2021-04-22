import torch
import torchvision.transforms as T
class CelebADataset():
    def __init__(self,dataset,size,zsize,debug=False):
        self.dataset=dataset
        self.zsize=zsize
        self.transforms=T.Compose([T.Resize(size),T.CenterCrop(size),T.ToTensor(),T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.len=len(self.dataset) if not debug else 12
    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        im,cls=self.dataset[idx]
        return torch.randn(self.zsize,1,1),self.transforms(im)
