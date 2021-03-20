import torch
import torch.nn as nn
import torch.nn.functional as F
import utils as U
import torchvision
from torchvision.utils import save_image
from model.dcgan import DCGAN
import core as Co
from dataset import CelebADataset
import os
import pickle as pkl
def operate():
    # fakemean=torch.zeros(2048)
    # fakesigma=torch.zeros(2048)
    # MVI=U.MeanVariance_iter()
    for i,(noise,realimg )in enumerate(loader):
        lossDreal,lossDfake,lossG,fake=M.trainbatch(noise.to(device),realimg.to(device))
        # fakesigma,fakemean=MVI.iter(fakesigma,fakemean,inception(fake.detach()))
        print(f'{e}/{epoch}:{i}/{len(loader)}, Dreal:{lossDreal:.2f}, Dfake:{lossDfake:.2f}, G:{lossG:.2f}')
        Co.addvalue(writer,'loss:Dreal',lossDreal,e)
        Co.addvalue(writer,'loss:Dfake',lossDfake,e)
        Co.addvalue(writer,'loss:G',lossG,e)
        if i == 0:
            save_image(((fake * 0.5) + 0.5), f'{savefolder}/{e}.png')
    # fid=U.fid(gtmean,gtsigma,fakemean,fakesigma)
    # IS=cal_is(realimg)
    # Co.addvalue(writer,'fid',fid,e)
    # Co.addvalue(writer,'IS',IS,e)

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
    parser.add_argument('--size',default=64,type=int)
    parser.add_argument('--loss',default='hinge')
    parser.add_argument('--feature',default=128,type=int)
    parser.add_argument('--cpu',default=False,action='store_true')
    args=parser.parse_args()
    epoch=args.epoch
    device='cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    savefolder='data/'+args.savefolder
    os.makedirs(savefolder,exist_ok=True)
    if args.checkpoint:
        chk=torch.load(args.checkpoint)
        loader=chk['loader']
        model=chk['model']
        e=chk['e']
        writer=chk['writer']
        args=chk['args']
    else:
        if args.loss=='hinge':
            def lossDreal(x):return F.relu(-x+1).mean()
            def lossDfake(x):return F.relu(x+1).mean()
            def lossG(x):return (-x).mean()
        elif args.loss=='bce':
            def lossDreal(x):return F.binary_cross_entropy(torch.sigmoid(x.reshape(-1)),torch.ones(x.shape[0],device=x.device))
            def lossDfake(x):return F.binary_cross_entropy(torch.sigmoid(x.reshape(-1)),torch.zeros(x.shape[0],device=x.device))
            def lossG(x):return F.binary_cross_entropy(torch.sigmoid(x.reshape(-1)),torch.ones(x.shape[0],device=x.device))
        elif args.loss=='mse':
            def lossDreal(x):return ((x-1)**2).mean()
            def lossDfake(x):return (x**2).mean()
            def lossG (x):return ((x-1)**2).mean()
        if args.dataset=='celeba':
            loader=torch.utils.data.DataLoader(CelebADataset(torchvision.datasets.CelebA('/opt/data','all',download=True),args.size,args.zsize),batch_size=args.batchsize,num_workers=4,shuffle=True)
        else:
            assert False,'celeba is allowed only.'
        if args.optimizer=='adam':
            optimizer=torch.optim.Adam
        if args.model == 'dcgan':
            model = DCGAN(optimizerG=optimizer,optimizerD=optimizer,lossDreal=lossDreal,lossDfake=lossDfake,lossG=lossG,zsize=args.zsize,feature=args.feature)
        writer={}
        e=0
    import json
    with open(f'{savefolder}/args.json','w') as f:
        json.dump(args.__dict__,f)
    inception=torchvision.models.inception_v3(pretrained=True)

    # if os.path.exists(inc_gt_outpath:=f'inception_{args.dataset}_{args.size}.pkl'):
    #     with open(inc_gt_outpath,'rb') as f:
    #         gtmean,gtsigma=pkl.load(f)
    # else:
    #     gtmean,gtsigma=U.make_gt_inception(inception,loader)
    #     with open(inc_gt_outpath,'wb') as f:
    #         pkl.dump([gtmean,gtsigma],f)
    M=model
    #TODO multi gpu
    if device=='cuda':
        model.discriminator=torch.nn.DataParallel(model.discriminator).to(device)
        model.generator=torch.nn.DataParallel(model.generator).to(device)
        # M=model.module
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
    Co.send_line_notify(f'{savefolder}/graphs.png',f'dcgan:{args.loss}')