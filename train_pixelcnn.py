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
from model.pixelcnn import PixelCNN
from utils.tfrecord import TFRDataloader
from torch.utils.tensorboard import SummaryWriter
import shutil


def operate(phase):
    if phase == 'train':
        model.train()
        loader = trainloader
    else:
        loader = valloader
    loader.init()

    mvci = U.MeanCoVariance_iter(device)

    for idx,img in enumerate(loader):
        iter_number[phase] += 1
        B, C, H, W = img.shape
        outstats = model.batch(img.to(device),phase=phase)
        log = f'{e}/{epoch}, {idx}/{len(loader)*0.75 if phase=="train" else 0.25},{outstats["loss"]}'
        with open(logpath, 'a')as f:
            f.write(log + '\n')
        print(log)
        writer.add_scalars('loss', outstats['loss'], iter_number[phase])

        if idx%1000==0:
            print('save image')
            generatedimages = model.generate(args.size,device=device,B=B)
            outstats['images']=outstats['images']*s+m
            generatedimages=generatedimages*s+m
            mvci.iter(inception(generatedimages.detach().to(device))[0])
            save_image(torch.cat([generatedimages[:B],outstats['images']],dim=2),f'{savefolder}/{e}_{idx}.jpg')
    writer.add_images('recon_images', outstats['images'], iter_number[phase])
    writer.add_images('gen_images', generatedimages, iter_number[phase])
    # get FID
    fid = U.fid(*inceptionstats[phase], *mvci.get(isbias=True))
    # IS=cal_is(realimg)
    writer.add_scalar(f'fid/{phase}', fid, iter_number[phase])
    # Co.addvalue(writer,'IS',IS,e)
    print(f'fid:{fid:.2f}')



if __name__ == '__main__':
    # try:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', default=32, type=int)
    parser.add_argument('--model', default='ae')
    parser.add_argument('--dataset', default='celeba')
    parser.add_argument('--optimizer', default='adam')
    parser.add_argument('--zsize', type=int, default=128)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--savefolder', default='tmp')
    parser.add_argument('--size', default=32, type=int)
    parser.add_argument('--reconloss', default='mse')
    parser.add_argument('--feature', default=384, type=int)
    parser.add_argument('--cpu', default=False, action='store_true')
    parser.add_argument('--datasetpath', default='../data')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--activation', default='relu')
    parser.add_argument('--disable_zviz', default=True, action='store_true')
    parser.add_argument('--KLD',default='Bernoulli')
    parser.add_argument('--dicsize',default=512,type=int)
    args = parser.parse_args()
    epoch = args.epoch
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    savefolder = 'data/' + args.savefolder
    from gtmodel import InceptionV3

    inception = InceptionV3([3]).to(device)
    logpath = f'{savefolder}/log.txt'
    shutil.rmtree(f'tfb/{args.savefolder}', ignore_errors=True)
    shutil.rmtree(f'{savefolder}', ignore_errors=True)
    writer = SummaryWriter(log_dir=f'tfb/{args.savefolder}')
    writer.add_text('args', f'{args}')
    os.makedirs(savefolder, exist_ok=True)

    if args.activation == 'relu':
        activation = nn.ReLU(inplace=True)
    elif args.activation == 'hswish':
        activation = nn.Hardswish(inplace=True)
    elif args.activation == 'lrelu':
        activation = nn.LeakyReLU(0.2, inplace=True)
    if args.reconloss == 'mse':
        reconloss = F.mse_loss
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam
    if args.model == 'ae':
        model = PixelCNN(feature=args.feature, activaton=activation, optimizer=optimizer)

    if args.dataset == 'celeba':
        # loader = torch.utils.data.DataLoader(
        #     CelebADataset(torchvision.datasets.CelebA(args.datasetpath, 'all', download=True), args.size, args.zsize,debug=args.debug),
        #     batch_size=args.batchsize, num_workers=4, shuffle=True)
        s,m=(0.5,0.5)
        trainloader = TFRDataloader(path=args.datasetpath + '/celeba.tfrecord', epoch=1, batch=args.batchsize,
                                    size=args.size, s=s,m=m, split='train')

        valloader = TFRDataloader(path=args.datasetpath + '/celeba.tfrecord', epoch=1, batch=args.batchsize,
                                  size=args.size, s=s,m=m, split='val')
    inceptionstats = {}
    for phase in ['train', 'val']:
        realstatspath = f'__{args.dataset}_real_{phase}.pkl'
        if os.path.exists(realstatspath):
            print('load real stats')
            with open(realstatspath, 'rb') as f:
                realsigma, realmu = pkl.load(f)
                realsigma = realsigma.to(device)
                realmu = realmu.to(device)
        else:
            print('make real stats')
            realsigma, realmu = U.make_gt_inception(inception, trainloader if phase == 'train' else valloader, device)
            with open(realstatspath, 'wb') as f:
                pkl.dump([realsigma.cpu(), realmu.cpu()], f)
        inceptionstats[phase] = [realsigma, realmu]
    with open(f'{savefolder}/args.json', 'w') as f:
        json.dump(args.__dict__, f)

    # if device == 'cuda':
    #     model = torch.nn.DataParallel(model).to(device)
    model=model.to(device)
    iter_number = {'train': 0, 'val': 0}
    # testinput = torch.randn(args.batchsize, 512, 2,2).to(device)
    testinput=torch.randint(0, args.dicsize, [args.batchsize , 2 , 2]).to(device)
    for e in range(epoch):
        operate('train')
        # operate('val')
    writer.close()

# except:
#     import traceback
#     import sys
#
#     error_msg = traceback.format_exc()
#     print(f'\033[31m{error_msg}\033[0m', file=sys.stderr)
