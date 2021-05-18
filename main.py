import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import torch
from torchvision.utils import save_image

torch.manual_seed(999)
import json
import pickle as pkl
import torch.nn.functional as F
import torch.nn as nn
import utils.util as U
from model.resnet import resnet18
from model.dcgan import DCGAN
from utils.tfrecord import TFRDataloader
from torch.utils.tensorboard import SummaryWriter
from model.common import SQLinear
import shutil


def operate():
    fakemvci = U.MeanCoVariance_iter(device)
    for i, realimg in enumerate(loader):

        B, C, H, W = realimg.shape
        noise = torch.randn(B, args.zsize, 1, 1)
        outstats = model.trainbatch(noise.to(device), realimg.to(device))
        fake = outstats['image']['fake']
        fakemvci.iter(inception(fake.detach().to(device))[0])
        stats = {'similarity': U.similarity(fake), 'rgb_distance': U.rgb_distance(fake)}
        log = f'{i},{outstats["loss"], stats}'
        with open(logpath, 'a')as f:
            f.write(log + '\n')
        print(log)
        writer.add_scalars('loss', outstats['loss'], i)
        writer.add_scalars('stats', stats, i)

        singularG = U.get_singular_with_SN(model.generator)
        singularD = U.get_singular_with_SN(model.discriminator)
        writer.add_scalars('singularG', singularG, i)
        writer.add_scalars('singularD', singularD, i)

        generatedimages = (model.generator(testinput) * 0.5) + 0.5
        save_image(generatedimages, f'{savefolder}/{i}.jpg')
        if i % 1000 == 0 and i != 0:

            writer.add_images('images', generatedimages, i)
            # get FID
            fid = U.fid(realsigma, realmu, *fakemvci.get(isbias=True))
            # IS=cal_is(realimg)
            writer.add_scalar('acc/fid', fid, i)
            # Co.addvalue(writer,'IS',IS,e)
            print(f'fid:{fid:.2f}')

            U.savecloudpickle({
                'model': model.to('cpu'),
                'e': e,
                'args': args,
                'realstats': (realsigma, realmu),
            }, savefolder + f'/chk.pth')
            model.to(device)


if __name__ == '__main__':
    # try:
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
        parser.add_argument('--debug', default=False, action='store_true')
        parser.add_argument('--g_activation', default='relu')
        parser.add_argument('--d_activation', default='lrelu')
        parser.add_argument('--disable_zviz', default=True, action='store_true')
        parser.add_argument('--discriminator', default=None)
        # parser.add_argument('--fakestatsper',default=10,type=int)
        args = parser.parse_args()
        epoch = args.epoch
        device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
        savefolder = 'data/' + args.savefolder
        from gtmodel import InceptionV3

        inception = InceptionV3([3]).to(device)
        e = 0
        logpath = f'{savefolder}/log.txt'
        if args.checkpoint:
            chk = U.loadcloudpickle(args.checkpoint)
            model = chk['model']
            model.zviz.clear()
            e = chk['e']
            args = chk['args']
            realsigma, realmu = chk['realstats']
        else:
            shutil.rmtree(f'tfb/{args.savefolder}', ignore_errors=True)
            shutil.rmtree(f'{savefolder}', ignore_errors=True)
        writer = SummaryWriter(log_dir=f'tfb/{args.savefolder}')
        os.makedirs(savefolder, exist_ok=True)
        if args.g_activation == 'relu':
            g_activation = nn.ReLU(inplace=True)
        elif args.g_activation == 'hswish':
            g_activation = nn.Hardswish(inplace=True)

        if args.d_activation == 'relu':
            d_activation = nn.ReLU(inplace=True)
        elif args.d_activation == 'hswish':
            d_activation = nn.Hardswish(inplace=True)
        elif args.d_activation == 'lrelu':
            d_activation = nn.LeakyReLU(0.2, inplace=True)

        if args.loss == 'hinge':
            lossDreal = lambda x: F.relu(-x + 1).mean()
            lossDfake = lambda x: F.relu(x + 1).mean()
            lossG = lambda x: (-x).mean()

        elif args.loss == 'sqhinge':
            lossDreal = lambda x: SQLinear()(F.relu(-x + 1)).mean()
            lossDfake = lambda x: SQLinear()(F.relu(x + 1)).mean()
            lossG = lambda x: SQLinear()(F.relu(-x + 1)).mean()
        elif args.loss == 'bce':
            lossDreal = lambda x: F.binary_cross_entropy_with_logits(x.reshape(-1),
                                                                     torch.ones(x.shape[0], device=x.device))
            lossDfake = lambda x: F.binary_cross_entropy_with_logits(x.reshape(-1),
                                                                     torch.zeros(x.shape[0], device=x.device))
            lossG = lambda x: F.binary_cross_entropy_with_logits(x.reshape(-1), torch.ones(x.shape[0], device=x.device))
        elif args.loss == 'mse':
            lossDreal = lambda x: ((x - 1) ** 2).mean()
            lossDfake = lambda x: (x ** 2).mean()
            lossG = lambda x: ((x - 1) ** 2).mean()
        elif args.loss == 'wgan':
            lossDreal = lambda x: torch.sigmoid(x).mean()
            lossDfake = lambda x: -torch.sigmoid(x).mean()
            lossG = lambda x: torch.sigmoid(x).mean()
        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam
        if args.discriminator is None:
            discriminator = args.discriminator
        elif args.discriminator == 'resnet18':
            discriminator = resnet18(activation=d_activation, num_classes=1)
        if args.model == 'dcgan':
            model = DCGAN(optimizerG=optimizer, optimizerD=optimizer, lossDreal=lossDreal, lossDfake=lossDfake,
                          lossG=lossG, zsize=args.zsize, feature=args.feature, d_activation=d_activation,
                          g_activation=g_activation, enable_zviz=not args.disable_zviz, discriminator=discriminator,
                          size=args.size)

        if args.dataset == 'celeba':
            # loader = torch.utils.data.DataLoader(
            #     CelebADataset(torchvision.datasets.CelebA(args.datasetpath, 'all', download=True), args.size, args.zsize,debug=args.debug),
            #     batch_size=args.batchsize, num_workers=4, shuffle=True)
            loader = TFRDataloader(path=args.datasetpath + '/celeba.tfrecord', epoch=epoch, batch=args.batchsize,
                                   size=args.size, s=0.5, m=0.5)
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
        # model=model.to(device)
        testinput = torch.randn(args.batchsize, args.zsize, 1, 1)
        operate()
        writer.close()

    # except:
    #     import traceback
    #     import sys
    #
    #     error_msg = traceback.format_exc()
    #     print(f'\033[31m{error_msg}\033[0m', file=sys.stderr)
