import torch
import torchvision.transforms as T
class CelebADataset():
    def __init__(self,dataset,size,zsize):
        self.dataset=dataset
        self.zsize=zsize
        self.transforms=T.Compose([T.Resize(size),T.CenterCrop(size),T.ToTensor(),T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    def __len__(self):
        return 12
        return len(self.dataset)
    def __getitem__(self, idx):
        im,cls=self.dataset[idx]
        return torch.randn(self.zsize,1,1),self.transforms(im)
