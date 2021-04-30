import torch

torch.manual_seed(999)
import json
import pickle as pkl
import torch.nn.functional as F
import torch.nn as nn
import utils.util as U
import torchvision
from torchvision.utils import save_image
from model.resnet import resnet18
from model.wrap import Wrap
from model.dcgan import DCGAN
import core as Co
from dataset import CelebADataset
import os
from utils.tfrecord import TFRDataloader


def operate():
    fakemvci = U.MeanCoVariance_iter(device)
    for i, realimg in enumerate(loader):
        B,C,H,W=realimg.shape
        noise=torch.randn(B,args.zsize,1,1)
        lossDreal, lossDfake, lossG, fake = model.trainbatch(noise.to(device), realimg.to(device))
        fakemvci.iter(inception(fake.detach().to(device))[0])
        log=f'{e}/{epoch}:{i}/{len(loader)}, Dreal:{lossDreal:.2f}, Dfake:{lossDfake:.2f}, G:{lossG:.2f}'
        with open(logpath,'a')as f:
            f.write(log+'\n')
        print(log)
        Co.addvalue(writer, 'loss:Dreal', lossDreal, e)
        Co.addvalue(writer, 'loss:Dfake', lossDfake, e)
        Co.addvalue(writer, 'loss:G', lossG, e)

        save_image(((model.generator(testinput) * 0.5) + 0.5), f'{savefolder}/{e}_{i}.png')
    Co.send_line_notify(f'{savefolder}/{e}_0.png', f'dcgan:{args.__dict__},{e}')
    # get FID
    fid = U.fid(realsigma, realmu, *fakemvci.get(isbias=True))
    # IS=cal_is(realimg)
    Co.addvalue(writer, 'acc:fid', fid, e)
    # Co.addvalue(writer,'IS',IS,e)
    print(f'fid:{fid:.2f}')



if __name__ == '__main__':
    try:
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument('--batchsize', default=32, type=int)
        parser.add_argument('--model', default='dcgan')
        parser.add_argument('--dataset', default='celeba')
        parser.add_argument('--optimizer', default='adam')
        parser.add_argument('--zsize', type=int, default=128)
        parser.add_argument('--epoch', default=100, type=int)
        parser.add_argument('--savefolder', default='tmp')
        parser.add_argument('--checkpoint', default=None)
        parser.add_argument('--size', default=64, type=int)
        parser.add_argument('--loss', default='hinge')
        parser.add_argument('--feature', default=128, type=int)
        parser.add_argument('--cpu', default=False, action='store_true')
        parser.add_argument('--datasetpath', default='../data')
        parser.add_argument('--debug',default=False,action='store_true')
        parser.add_argument('--g_activation',default='relu')
        parser.add_argument('--d_activation',default='relu')
        parser.add_argument('--disable_zviz',default=False,action='store_true')
        parser.add_argument('--discriminator',default=None)
        # parser.add_argument('--fakestatsper',default=10,type=int)
        args = parser.parse_args()
        epoch = args.epoch
        device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
        savefolder = 'data/' + args.savefolder
        os.makedirs(savefolder, exist_ok=True)
        from gtmodel import InceptionV3

        inception = InceptionV3([3]).to(device)

        writer = {}
        e = 0
        logpath=f'{savefolder}/log.txt'
        if args.checkpoint:
            chk = U.loadcloudpickle(args.checkpoint)
            model = chk['model']
            model.zviz.clear()
            e = chk['e']
            writer = chk['writer']
            args = chk['args']
            realsigma, realmu = chk['realstats']
        else:
            if os.path.exists(logpath):
                os.remove(logpath)
        if args.g_activation=='relu':
            g_activation=nn.ReLU(inplace=True)
        elif args.g_activation=='hswish':
            g_activation=nn.Hardswish(inplace=True)

        if args.d_activation=='relu':
            d_activation=nn.ReLU(inplace=True)
        elif args.d_activation=='hswish':
            d_activation=nn.Hardswish(inplace=True)

        if args.loss == 'hinge':
            lossDreal =lambda x:F.relu(-x + 1).mean()
            lossDfake =lambda x: F.relu(x + 1).mean()
            lossG =lambda x:(-x).mean()
        elif args.loss == 'bce':
            lossDreal =lambda x:F.binary_cross_entropy_with_logits(x.reshape(-1), torch.ones(x.shape[0], device=x.device))
            lossDfake =lambda x:F.binary_cross_entropy_with_logits(x.reshape(-1), torch.zeros(x.shape[0], device=x.device))
            lossG =lambda x:F.binary_cross_entropy_with_logits(x.reshape(-1), torch.ones(x.shape[0], device=x.device))
        elif args.loss == 'mse':
            lossDreal =lambda x:((x - 1) ** 2).mean()
            lossDfake =lambda x: (x ** 2).mean()
            lossG= lambda x:((x - 1) ** 2).mean()
        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam
        if args.discriminator is None:
            discriminator=args.discriminator
        elif args.discriminator=='resnet18':
            discriminator=resnet18(activation=d_activation,num_classes=1)
        if args.model == 'dcgan':
            model = DCGAN(optimizerG=optimizer, optimizerD=optimizer, lossDreal=lossDreal, lossDfake=lossDfake,
                          lossG=lossG, zsize=args.zsize, feature=args.feature,d_activation=d_activation,g_activation=g_activation,enable_zviz=not args.disable_zviz,discriminator=discriminator)


        if args.dataset == 'celeba':
            # loader = torch.utils.data.DataLoader(
            #     CelebADataset(torchvision.datasets.CelebA(args.datasetpath, 'all', download=True), args.size, args.zsize,debug=args.debug),
            #     batch_size=args.batchsize, num_workers=4, shuffle=True)
            loader=TFRDataloader(path=args.datasetpath+'/celeba.tfrecord',epoch=1,batch=args.batchsize,size=args.size)
        realstatspath = f'__{args.dataset}_real.pkl'
        if os.path.exists(realstatspath):
            print('load real stats')
            with open(realstatspath, 'rb') as f:
                realsigma, realmu = pkl.load(f)
                realsigma = realsigma.to(device)
                realmu = realmu.to(device)
        else:
            print('make real stats')
            realsigma, realmu = U.make_gt_inception(inception, loader, device)
            with open(realstatspath, 'wb') as f:
                pkl.dump([realsigma.cpu(), realmu.cpu()], f)
        with open(f'{savefolder}/args.json', 'w') as f:
            json.dump(args.__dict__, f)

        if device == 'cuda':
            model.discriminator = torch.nn.DataParallel(model.discriminator).to(device)
            model.generator = torch.nn.DataParallel(model.generator).to(device)
            # M=model.module
        testinput=torch.randn(args.batchsize,args.zsize,1,1)
        for e in range(e, epoch):
            operate()

            U.savecloudpickle({
                'model': model.to('cpu'),
                'e': e,
                'writer': writer,
                'args': args,
                'realstats': (realsigma, realmu),
            }, savefolder +f'/chk.pth')
            model.to(device)
            Co.savedic(writer, savefolder)
            Co.send_line_notify(f'{savefolder}/graphs.png', f'dcgan:{args.__dict__},{e}')

    except:
        import traceback
        import sys
        error_msg=traceback.format_exc()
        print(f'\033[31m{error_msg}\033[0m',file=sys.stderr)
