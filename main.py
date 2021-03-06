import torch
import torch.nn as nn
import torch.nn.functional as F
import utils as U
import torchvision
from torchvision.utils import save_image
from model.dcgan import DCGAN
import core as Co
from dataset import CelebADataset

def operate():
    for i,(noise,realimg )in enumerate(loader):
        lossDreal,lossDfake,lossG,fake=M.trainbatch(noise.to(device),realimg.to(device))

        # fid=cal_fid(realimg)
        # IS=cal_is(realimg)
        print(f'{e}/{epoch}:{i}/{len(loader)}, Dreal:{lossDreal:.2f}, Dfake:{lossDfake:.2f}, G:{lossG:.2f}')
        Co.addvalue(writer,'loss:Dreal',lossDreal,e)
        Co.addvalue(writer,'loss:Dfake',lossDfake,e)
        Co.addvalue(writer,'loss:G',lossG,e)
        # Co.addvalue(writer,'fid',fid,e)
        # Co.addvalue(writer,'IS',IS,e)
        if i==0:
            save_image((fake*0.5)+0.5,f'{savefolder}/{e}.png')

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--batchsize',default=8,type=int)
    parser.add_argument('--model',default='dcgan')
    parser.add_argument('--dataset',default='celeba')
    parser.add_argument('--optimizer',default='adam')
    parser.add_argument('--zsize',type=int,default=128)
    parser.add_argument('--epoch',default=100,type=int)
    parser.add_argument('--savefolder',default='tmp')
    parser.add_argument('--checkpoint',default=None)
    parser.add_argument('--size',default=128,type=int)
    args=parser.parse_args()
    epoch=args.epoch
    device='cuda' if torch.cuda.is_available() else 'cpu'
    savefolder='data/'+args.savefolder
    import os
    os.makedirs(savefolder,exist_ok=True)
    if args.checkpoint:
        chk=torch.load(args.checkpoint)
        loader=chk['loader']
        model=chk['model']
        e=chk['e']
        writer=chk['writer']
        args=chk['args']
    else:
        # lossDreal=lambda x:-U.min(x-1,0)
        # lossDfake=lambda x:-U.min(-x-1,0)
        # lossG=lambda x:-x
        def lossDreal(x):return (x-1)**2
        def lossDfake(x):return x**2
        def lossG (x):return (x-1)**2
        if args.dataset=='celeba':
            loader=torch.utils.data.DataLoader(CelebADataset(torchvision.datasets.CelebA('/opt/data','all',download=True),args.size,args.zsize),batch_size=args.batchsize,num_workers=4,shuffle=True)
        if args.optimizer=='adam':
            optimizer=torch.optim.Adam
        if args.model == 'dcgan':
            model = DCGAN(optimizerG=optimizer,optimizerD=optimizer,lossDreal=lossDreal,lossDfake=lossDfake,lossG=lossG,zsize=args.zsize)
        writer={}
        e=0
    import json
    with open(f'{savefolder}/args.json','w') as f:
        json.dump(args.__dict__,f)
    M=model
    #TODO multi gpu
    if device=='cuda':
        model=torch.nn.DataParallel(model).to(device)
        M=model.module
    for e in range(e,epoch):
        operate()
        torch.save({
            'model':model.to('cpu'),
            'e':e,
            'writer':writer,
            'args':args,
            'loader':loader
        },savefolder+'/chk.pth')
        model.to(device)
        Co.savedic(writer,savefolder,"")