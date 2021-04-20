import torch
import utils.util as U
import glob
import torchvision.transforms as T
from PIL import Image
import pytorch_fid.fid_score
from model.inception import inception_v3
class FileDataset:
    def __init__(self,path):
        self.imgs=glob.glob(f'{path}/*.jpg')
        self.transforms=T.Compose([
            # T.Resize(299),
            T.ToTensor()
        ])
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, idx):
        return 0,self.transforms(Image.open(self.imgs[idx]).convert('RGB'))
batchsize=64
loader=torch.utils.data.DataLoader(FileDataset('/home/hokusei/Downloads/testSet/miniSet'),batch_size=batchsize,num_workers=4)
loader2=torch.utils.data.DataLoader(FileDataset('/home/hokusei/Downloads/testSet/miniSet'),batch_size=batchsize,num_workers=4)
device='cuda' if torch.cuda.is_available() else 'cpu'
# device='cpu'
# model=inception_v3(pretrained=True,aux_logits=False).to(device)
from gtmodel import InceptionV3
model=InceptionV3([3]).to(device)
model.eval()
cs1,m1=U.make_gt_inception(model,loader,device=device)
cs2,m2=U.make_gt_inception(model,loader2,device=device)
fid=U.fid(cs1,m1,cs2,m2)
print(fid)
