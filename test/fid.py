import torch
import utils.util as U
import glob
import torchvision.transforms as T
from PIL import Image
from model.inception import inception_v3
class FileDataset:
    def __init__(self,path):
        self.imgs=glob.glob(f'{path}/*.jpg')
        self.transforms=T.Compose([T.Resize(299),T.ToTensor()])
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, idx):
        return 0,self.transforms(Image.open(self.imgs[idx])).repeat(3,1,1)

loader=torch.utils.data.DataLoader(FileDataset('/home/hokusei/Downloads/testSet/testSet'),batch_size=8,num_workers=4)
loader2=torch.utils.data.DataLoader(FileDataset('/home/hokusei/Downloads/testSet/testSet2'),batch_size=8,num_workers=4)
model=inception_v3(pretrained=True)
device='cuda' if torch.cuda.is_available() else 'cpu'
m1,s1=U.make_gt_inception(model,loader,device=device)
m2,s2=U.make_gt_inception(model,loader2,device=device)
fid=U.fid(s1,m1,s2,m2)
print(fid)
