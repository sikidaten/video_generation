from __future__ import print_function

import os
# %matplotlib inline
import random
import shutil
from functools import partial

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Root directory for dataset
folder = "TUTORIAL"
dataroot = "../data/celeba/"
savefolder = f'data/{folder}/'
tfbfolder = f'tfb/{folder}/'
if os.path.exists(savefolder): shutil.rmtree(savefolder)
if os.path.exists(tfbfolder): shutil.rmtree(tfbfolder)
os.mkdir(savefolder)
os.mkdir(tfbfolder)
# Number of workers for dataloader
workers = 4

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# We can use an image folder dataset the way we have it setup.
# Create the dataset
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# Decide which device we want to run on
device = torch.device("cuda")

# Plot some training images
real_batch = next(iter(dataloader))


# plt.figure(figsize=(8,8))
# plt.axis("off")
# plt.title("Training Images")
# plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Generator Code

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


# Create the generator
netG = Generator(ngpu).to(device)
# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netG.apply(weights_init)

# Print the model
print(netG)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, input):
        return self.main(input)


# Create the Discriminator
netD = Discriminator(ngpu).to(device)

# netD=resnet18(num_classes=1).to(device)
# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)

# Print the model
print(netD)


def savegrad(self, gradinput, gradoutput, modulename, thres=0.01):
    global i_grad
    i_grad += 1
    with torch.no_grad():
        size = 64
        gout = gradoutput[0]
        if gout.max().abs() > thres: graddic[f'{modulename}:{i_grad}:max' + self.__class__.__name__] = gout.max().item()
        if gout.mean().abs() > thres: graddic[
            f'{modulename}:{i_grad}:mean' + self.__class__.__name__] = gout.mean().item()
        if gout.min().abs() > thres: graddic[f'{modulename}:{i_grad}:min' + self.__class__.__name__] = gout.min().item()
        print(f'{modulename}:{i_grad}:{self.__class__.__name__}')
        gout = (gout - gout.min()) / (gout.max() - gout.min())
        gout = gout.abs().max(dim=0)[0].max(dim=0)[0].unsqueeze(0).unsqueeze(0)
        img = F.interpolate(gout, size=(size, size)).squeeze(0)
        gradimgs.append(img)


for m_name, modules in zip(['D', 'G'], [netD.named_modules(), netG.named_modules()]):
    for name, module in modules:
        print(module.__class__.__name__)
        if module.__class__.__name__ in ['Tanh', 'ConvTranspose2d', 'ReLU', 'BatchNorm2d', 'Conv2d', 'LeakyReLU']:
            module.register_backward_hook(partial(savegrad, modulename=m_name))
# Initialize BCELoss function
criterion = lambda x, y: F.binary_cross_entropy(F.sigmoid(x), y)

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
writer = SummaryWriter(log_dir=tfbfolder)
print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):
        gradimgs = []
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        graddic = {}
        i_grad = 0
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()
        writer.add_scalars('Dreal', graddic, epoch * len(dataloader) + i)

        ## Train with all-fake batch
        # Generate batch of latent vectors
        graddic = {}
        i_grad = 0
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        writer.add_scalars('Dfake', graddic, epoch * len(dataloader) + i)
        if i % 10 == 0:
            dic = {}
            for name, p in netD.named_parameters():
                if p.grad.max().abs() > 0.1: dic[f'{name}:max'] = p.grad.max().item()
                if p.grad.mean().abs() > 0.1: dic[f'{name}:mean'] = p.grad.mean().item()
                if p.grad.min().abs() > 0.1: dic[f'{name}:min'] = p.grad.min().item()
            writer.add_scalars('D_grad', dic, epoch * len(dataloader) + i)
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        graddic = {}
        i_grad = 0
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        writer.add_scalars('G', graddic, epoch * len(dataloader) + i)
        if i % 10 == 0:
            dic = {}
            for name, p in netG.named_parameters():
                if p.grad.max().abs() > 0.1: dic[f'{name}:max'] = p.grad.max().item()
                if p.grad.mean().abs() > 0.1: dic[f'{name}:mean'] = p.grad.mean().item()
                if p.grad.min().abs() > 0.1: dic[f'{name}:min'] = p.grad.min().item()
            writer.add_scalars('G_grad', dic, epoch * len(dataloader) + i)
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
              % (epoch, num_epochs, i, len(dataloader),
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
        save_image((fake + 1) / 2, f'{savefolder}/{epoch}_{i}.jpg')
        save_image(torch.stack(gradimgs), f'{savefolder}/grad_{epoch}_{i}.jpg')

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        writer.add_scalars('loss', {'Dreal': errD_real, 'Dfake': errD_fake, 'G': errG},
                           global_step=epoch * len(dataloader) + i)
writer.close()
