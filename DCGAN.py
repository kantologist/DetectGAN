import argparse
import os
import numpy as np
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
import vad as V



cuda = True if torch.cuda.is_available() else False


def my_collate(batch_feats):
    average_len = 500#sum([len(utt) for utt in batch_feats]) // len(batch_feats)
    feats = np.zeros((len(batch_feats),average_len,batch_feats[0].shape[1]))
    for i,feat in enumerate(batch_feats):
        batch = V.VAD(feat)
        if len(batch) < average_len:
            padded = np.pad(batch,((0,average_len-len(batch)),(0,0)),mode='wrap')
            feats[i] = padded
        else:
            feats[i] = batch[:average_len,:]
        
    return torch.from_numpy(feats).unsqueeze(1)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Discriminator(nn.Module):
    def __init__(
            self, num_gpu
            ):

        super(Discriminator, self).__init__()
        self.num_gpu = num_gpu        

        self.main = nn.Sequential(
                    nn.Conv2d(1, 32, 4, 2, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(32, 64, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.LeakyReLU(0.2, inplace=True),

                    nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(64 * 2),
                    nn.LeakyReLU(0.2, inplace=True),

                    nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(64 * 4),
                    nn.LeakyReLU(0.2, inplace=True),

                    nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(64 * 8),
                    nn.LeakyReLU(0.2, inplace=True),

                    nn.AdaptiveMaxPool2d(1),
                    nn.Conv2d(512,64,kernel_size=(1,1)),
                    nn.Conv2d(64,8,kernel_size=(1,1)),
                    nn.Conv2d(8,1,kernel_size=(1,1)),
                    nn.Sigmoid()
                    )

    def forward(self, input):
        return self.main( input )
 
# Varied Input Generator
class Generator(nn.Module):
    def __init__(self,num_gpu,):
        super(Generator, self).__init__()
        self.num_gpu = num_gpu
               
        self.conv1 = nn.Conv2d(1, 64, 4, 2, 1, bias=False)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64 * 2)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

	# 64x64 to 32x32
        self.conv3 = nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(64 * 4)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)

	# 32x32 to 16x16
        self.conv4 = nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(64 * 8)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)

	# 16x16 to 8x8
        self.conv5 = nn.Conv2d(64 * 8, 64 * 8, 4, 2, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(64 * 8)
        self.relu5 = nn.LeakyReLU(0.2, inplace=True)


        # Varied length feature inside (8x8 to 4x4)
        self.conv6 = nn.Conv2d(64 * 8, 64 * 8, 4, 2, 1, bias=False)
        self.bn6 = nn.BatchNorm2d(64 * 8)
        self.relu6 = nn.LeakyReLU(0.2, inplace=True)

        # 4x4 to 8x8
        self.tconv6 = nn.ConvTranspose2d(64 * 8, 64 * 8, 4, 2, 1, bias=False) 
        self.tbn6 = nn.BatchNorm2d(64 * 8) 
        self.trelu6 = nn.ReLU(True) 

            # 8x8 to 16x16
        self.tconv5 = nn.ConvTranspose2d(64 * 8, 64 * 8, 4, 2, 1, bias=False) 
        self.tbn5 = nn.BatchNorm2d(64 * 8) 
        self.trelu5 = nn.ReLU(True) 

            # 16x16 to 32x32
        self.tconv4 = nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False) 
        self.tbn4 = nn.BatchNorm2d(64 * 4) 
        self.trelu4 = nn.ReLU(True) 

            # 32x32 to 64X64
        self.tconv3 = nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False) 
        self.tbn3 = nn.BatchNorm2d(64 * 2) 
        self.trelu3 = nn.ReLU(True) 

            # 64x64 to 128X128
        self.tconv2 = nn.ConvTranspose2d(64 * 2,     64, 4, 2, 1, bias=False) 
        self.tbn2 = nn.BatchNorm2d(64) 
        self.trelu2 = nn.ReLU(True) 

            # 128x128 to 256X256
        self.tconv1 = nn.ConvTranspose2d(    64,      1, 4, 2, 1, bias=False) 


    def forward(self, input):
        conv1 = self.conv1( input )
        relu1 = self.relu1( conv1 )

        conv2 = self.conv2( relu1 )
        bn2 = self.bn2( conv2 )
        relu2 = self.relu2( bn2 )
       
        conv3 = self.conv3( relu2 )
        bn3 = self.bn3( conv3 )
        relu3 = self.relu3( bn3 )

        conv4 = self.conv4( relu3 )
        bn4 = self.bn4( conv4 )
        relu4 = self.relu4( bn4 )

        conv5 = self.conv5( relu4 )
        bn5 = self.bn5( conv5 )
        relu5 = self.relu5( bn5 )

        conv6 = self.conv6( relu5 )
        bn6 = self.bn6( conv6 )
        relu6 = self.relu6( bn6 )

        ## Transposed CNN
 
        tconv6 = self.tconv6(relu6)
        tbn6 = self.tbn6( tconv6 )
        trelu6 = self.trelu6(tbn6)

        tconv5 = self.tconv5(trelu6)
        tbn5 = self.tbn5(tconv5) 
        trelu5 = self.trelu5(tbn5) 

        tconv4 = self.tconv4(trelu5) 
        tbn4 = self.tbn4(tconv4)
        trelu4 = self.trelu4(tbn4) 

        tconv3 = self.tconv3(trelu4) 
        tbn3 = self.tbn3(tconv3) 
        trelu3 = self.trelu3(tbn3) 

        tconv2 = self.tconv2(trelu3)
        tbn2 = self.tbn2(tconv2)
        trelu2 = self.trelu2(tbn2)

        tconv1 = self.tconv1(trelu2)

        # pdb.set_trace()
        return torch.sigmoid( tconv1 ), [relu1, relu2, relu3, relu4, relu5], [trelu2, trelu3, trelu4, trelu5, trelu6]

def main(argv):
    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    # Initialize generator and discriminator
    generator = Generator(1)
    discriminator = Discriminator(1)

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
    
    #generator.load_state_dict(torch.load("models/generator43.pt"))
    discriminator.load_state_dict(torch.load("models/discriminator43.pt"))

    # Configure data loader
    os.makedirs('../../data/mnist', exist_ok=True)
    
    dataloader = torch.utils.data.DataLoader((np.load('male_spect.npy')),batch_size=opt.batch_size, 
                                             shuffle=True, collate_fn = my_collate)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # ----------
    #  Training
    # ----------

    for epoch in range(opt.n_epochs):
        d_losses = []
        g_losses = []
        d_loss = 0
        for i, (imgs) in enumerate(dataloader):
            valid = Variable(Tensor(imgs.shape[0]).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.shape[0]).fill_(0.0), requires_grad=False)
            real_imgs = Variable(imgs.type(Tensor))
            optimizer_G.zero_grad()
            #z = Variable(Tensor(np.random.random(imgs.shape)))
            gen_imgs,_,_ = generator(real_imgs)
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()
            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            
            d_loss.backward()
            optimizer_D.step()

            
            
            d_losses.append(d_loss.data.cpu().numpy())
            g_losses.append(g_loss.data.cpu().numpy())
            if i % 5 == 0:
                #d_loss.backward()
                #optimizer_D.step()
                #d_loss = 0
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, opt.n_epochs, i, len(dataloader),
                                                                np.mean(d_losses), np.mean(g_losses)))
        torch.save(generator.state_dict(), "models/generator"+str(epoch)+".pt")
        torch.save(discriminator.state_dict(), "models/discriminator"+str(epoch)+".pt")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=32, help='size of the batches')
    parser.add_argument('--lr', type=float, default=0.001, help='adam: learning rate')
    parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
    parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
    parser.add_argument('--channels', type=int, default=1, help='number of image channels')
    parser.add_argument('--sample_interval', type=int, default=400, help='interval between image sampling')
    opt = parser.parse_args()

    main(sys.argv)
