## New Model Run Script Date: Oct 8th
## Author: Yang Gao

import time
import os
import fnmatch
import argparse
from itertools import chain
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from preprocessing import *
from model_Adp import *
import scipy
import scipy.io as sio
import numpy as np
from progressbar import ETA, Bar, Percentage, ProgressBar
import pdb
import librosa
from sklearn import preprocessing
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser(description='PyTorch implementation of DiscoGAN')
parser.add_argument('--num_gpu', type=int, default=1) ## add num_gpu
parser.add_argument('--delta', type=str, default='true', help='Set to use or not use delta feature')
parser.add_argument('--cuda', type=str, default='true', help='Set cuda usage')
parser.add_argument('--task_name', type=str, default='spectrogram', help='Set data name')
parser.add_argument('--epoch_size', type=int, default=2000, help='Set epoch size')
parser.add_argument('--batch_size', type=int, default=8, help='Set batch size')
parser.add_argument('--learning_rate', type=float, default=0.0002, help='Set learning rate for optimizer')
parser.add_argument('--result_path', type=str, default='./results/', help='Set the path the result images will be saved.')
parser.add_argument('--model_path', type=str, default='./models/', help='Set the path for trained models')
parser.add_argument('--model_arch', type=str, default='spec_gan', help='choose among gan/recongan/discogan/spec_gan. gan - standard GAN, recongan - GAN with reconstruction, discogan - DiscoGAN, spec_gan - My modified GAN model for speech.')
parser.add_argument('--image_size', type=int, default=256, help='Image size. 64 for every experiment in the paper')

parser.add_argument('--gan_curriculum', type=int, default=1000, help='Strong GAN loss for certain period at the beginning')
parser.add_argument('--starting_rate', type=float, default=0.01, help='Set the lambda weight between GAN loss and Recon loss during curriculum period at the beginning. We used the 0.01 weight.')
parser.add_argument('--default_rate', type=float, default=0.5, help='Set the lambda weight between GAN loss and Recon loss after curriculum period. We used the 0.5 weight.')

parser.add_argument('--n_test', type=int, default=20, help='Number of test data.')

parser.add_argument('--update_interval', type=int, default=10, help='') # origin 3

parser.add_argument('--log_interval', type=int, default=10, help='Print loss values every log_interval iterations.')
parser.add_argument('--image_save_interval', type=int, default=2000, help='Save test results every image_save_interval iterations.')
parser.add_argument('--model_save_interval', type=int, default=10000, help='Save models every model_save_interval iterations.')

def as_np(data):
    return data.cpu().data.numpy()

def get_data():
    
    male_spect = np.load('male_spect_1000.npy')
    data_A = male_spect[:int(len(male_spect)*.8)]
    test_A = male_spect[int(len(male_spect)*.8)+1:]
    
    female_spect = np.load('female_spect_1000.npy')
    data_B = female_spect[:int(len(female_spect)*.8)]
    test_B = female_spect[int(len(female_spect)*.8)+1:]

    return data_A, data_B, test_A, test_B

def get_fm_loss(real_feats, fake_feats, criterion):
	losses = 0
	for real_feat, fake_feat in zip(real_feats, fake_feats):
	# pdb.set_trace()
		l2 = (real_feat.mean(0) - fake_feat.mean(0)) * (real_feat.mean(0) - fake_feat.mean(0))
		loss = criterion( l2, Variable( torch.ones( l2.size() ) ).cuda() )
		losses += loss

	return losses

## Change to 3 inputs 
def get_gan_loss(dis_real, dis_fake1, dis_fake2, criterion, cuda):
	labels_dis_real = Variable(torch.ones( [dis_real.size()[0], 1] ))
	labels_dis_fake1 = Variable(torch.zeros([dis_fake1.size()[0], 1] ))
	labels_dis_fake2 = Variable(torch.zeros([dis_fake2.size()[0], 1] ))
	labels_gen1 = Variable(torch.ones([dis_fake1.size()[0], 1]))
	labels_gen2 = Variable(torch.ones([dis_fake2.size()[0], 1]))

	if cuda:
		labels_dis_real = labels_dis_real.cuda()
		labels_dis_fake1 = labels_dis_fake1.cuda()
		labels_dis_fake2 = labels_dis_fake2.cuda()
		labels_gen1 = labels_gen1.cuda()
		labels_gen2 = labels_gen2.cuda()

	dis_loss = criterion( dis_real, labels_dis_real ) * 0.4 + criterion( dis_fake1, labels_dis_fake1 ) * 0.3 + criterion( dis_fake2, labels_dis_fake2 ) * 0.3
	gen_loss = criterion( dis_fake1, labels_gen1 ) * 0.5 + criterion( dis_fake2, labels_gen2 ) * 0.5

	return dis_loss, gen_loss

## Use CrossEntropyLoss: target should be N
def get_stl_loss(A_stl, A1_stl, A2_stl, B_stl, B1_stl, B2_stl, criterion, cuda):
	# for nn.CrossEntropyLoss, the target is class index.
	labels_A = Variable(torch.ones( A_stl.size()[0] )) # NLL/CE target N not Nx1
	labels_A.data =  labels_A.data.type(torch.LongTensor)

	labels_A1 = Variable(torch.ones( A1_stl.size()[0] )) # NLL/CE target N not Nx1
	labels_A1.data =  labels_A1.data.type(torch.LongTensor)

	labels_A2 = Variable(torch.ones( A2_stl.size()[0] )) # NLL/CE target N not Nx1
	labels_A2.data =  labels_A2.data.type(torch.LongTensor)

	labels_B = Variable(torch.zeros(B_stl.size()[0] ))
	labels_B.data =  labels_B.data.type(torch.LongTensor)

	labels_B1 = Variable(torch.zeros(B1_stl.size()[0] ))
	labels_B1.data =  labels_B1.data.type(torch.LongTensor)

	labels_B2 = Variable(torch.zeros(B2_stl.size()[0] ))
	labels_B2.data =  labels_B2.data.type(torch.LongTensor)

	if cuda:
		labels_A = labels_A.cuda()
		labels_A1 = labels_A1.cuda()
		labels_A2 = labels_A2.cuda()
		labels_B = labels_B.cuda()
		labels_B1 = labels_B1.cuda()
		labels_B2 = labels_B2.cuda()

	A_stl = np.squeeze(A_stl)
	A1_stl = np.squeeze(A1_stl)
	A2_stl = np.squeeze(A2_stl)
	B_stl = np.squeeze(B_stl)
	B1_stl = np.squeeze(B1_stl)
	B2_stl = np.squeeze(B2_stl)

	stl_loss_A = criterion( A_stl, labels_A ) * 0.2 + criterion( A1_stl, labels_A1 ) * 0.15 + criterion( A2_stl, labels_A2 ) * 0.15
	stl_loss_B = criterion( B_stl, labels_B ) * 0.2 + criterion( B1_stl, labels_B1 ) * 0.15 + criterion( B2_stl, labels_B2 ) * 0.15
	stl_loss = stl_loss_A + stl_loss_B

	return stl_loss

def delta_regu(input_v, batch_size, criterion=nn.MSELoss()):
	losses = 0
	for i in range(batch_size):
		# pdb.set_trace()
		input_temp = np.squeeze(input_v.data[i,:,:,:])
		# no need to take mean among 3 channels since current input is 256x256 instead of 3x256x256
		# input_temp = np.mean(input_temp.cpu().numpy(), axis = 0)
		input_temp = input_temp.cpu().numpy()
		input_delta = np.absolute(librosa.feature.delta(input_temp))
		b=input_delta.shape[1]
		delta_loss = criterion(Variable((torch.from_numpy(input_delta)).type(torch.DoubleTensor)), Variable((torch.zeros([256,b])).type(torch.DoubleTensor)))
		# delta_loss = criterion((torch.from_numpy(input_delta)), Variable((torch.zeros([256,256]))))
		losses += delta_loss

	delta_losses = losses/batch_size

	return delta_losses.type(torch.cuda.FloatTensor)  

def normf(A):
	x = A.data.cpu().numpy()
	x_min = x.min(axis=(0, 1), keepdims=True)
	x_max = x.max(axis=(0, 1), keepdims=True)
	x = (x - x_min)/(x_max-x_min)
	x = Variable((torch.from_numpy(x)).type(torch.FloatTensor))
	return x

def my_collate(batch_feats):
   
    average_len = 500#sum([len(utt) for utt in batch_feats]) // len(batch_feats)
    feats = np.zeros((len(batch_feats),average_len,batch_feats[0].shape[1]))
    for i,batch in enumerate(batch_feats):
        if len(batch) < average_len:
            padded = np.pad(batch,((0,average_len-len(batch)),(0,0)),mode='wrap')
            feats[i] = padded
        else:
            feats[i] = batch[:average_len,:]
        
    return torch.from_numpy(feats).unsqueeze(1).float()

if __name__ == '__main__':


    global args, data_A
    args = parser.parse_args()


    cuda = args.cuda
    if cuda == 'true':
        cuda = True
    else:
        cuda = False

    task_name = args.task_name

    epoch_size = args.epoch_size
    batch_size = args.batch_size

    result_path = os.path.join( args.result_path, args.task_name )
    result_path = os.path.join( result_path, args.model_arch )

    model_path = os.path.join( args.model_path, args.task_name )
    model_path = os.path.join( model_path, args.model_arch )

    data_style_A, data_style_B, test_style_A, test_style_B = get_data()
    

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    generator_A = Generator(args.num_gpu)
    generator_B = Generator(args.num_gpu)
    discriminator_A = Discriminator(args.num_gpu)
    discriminator_B = Discriminator(args.num_gpu)
    discriminator_S = StyleDiscriminator(args.num_gpu)

    if cuda:
        generator_A = generator_A.cuda()
        generator_B = generator_B.cuda()
        discriminator_A = discriminator_A.cuda()
        discriminator_B = discriminator_B.cuda()
        discriminator_S = discriminator_S.cuda()

    if args.num_gpu > 1:
        # test_A_V = nn.DataParallel(test_A_V, device_ids = range(args.num_gpu))
        # test_B_V = nn.DataParallel(test_B_V, device_ids = range(args.num_gpu))
        generator_A = nn.DataParallel(generator_A, device_ids = range(args.num_gpu))
        generator_B = nn.DataParallel(generator_B, device_ids = range(args.num_gpu))
        discriminator_A = nn.DataParallel(discriminator_A, device_ids = range(args.num_gpu))
        discriminator_B = nn.DataParallel(discriminator_B, device_ids = range(args.num_gpu))
        discriminator_S = nn.DataParallel(discriminator_S, device_ids = range(args.num_gpu)) 

    #data_size = min( len(data_style_A), len(data_style_B) )
    #n_batches = ( data_size // batch_size )

    recon_criterion = nn.L1Loss() #MSELoss()
    gan_criterion = nn.BCELoss()
    feat_criterion = nn.HingeEmbeddingLoss()
    stl_criterion = nn.CrossEntropyLoss()

    gen_params = chain(generator_A.parameters(), generator_B.parameters())
    dis_params = chain(discriminator_A.parameters(), discriminator_B.parameters())
    stl_params =  discriminator_S.parameters() 

    optim_gen = optim.Adam( gen_params, lr=args.learning_rate, betas=(0.5,0.999), weight_decay=0.00001)
    optim_dis = optim.Adam( dis_params, lr=args.learning_rate, betas=(0.5,0.999), weight_decay=0.00001)
    optim_stl = optim.Adam( stl_params, lr=args.learning_rate, betas=(0.5,0.999), weight_decay=0.00001)

    iters = 0
    start = time.time()    

    log_gen_loss = []
    log_dis_loss = []
    log_stl_loss = []
    log_delta_A = []
    log_delta_B = []
    log_fm_loss_A = []
    log_fm_loss_B = []
    log_recon_loss_A = []
    log_recon_loss_B = [] 
    log_gen_loss_A = []
    log_gen_loss_B = []
    
    batch_size = 1
    
    A_loader = DataLoader( data_style_A, batch_size=batch_size , 
                          shuffle=True, collate_fn = my_collate)
    B_loader = DataLoader( data_style_B, batch_size=batch_size , 
                          shuffle=True, collate_fn = my_collate)
    A_test_loader = DataLoader( test_style_A, batch_size=batch_size , 
                               shuffle=True, collate_fn = my_collate)
    B_test_loader = DataLoader( test_style_B, batch_size=batch_size , 
                               shuffle=True, collate_fn = my_collate)
    
    data_size = min( len(data_style_A), len(data_style_B) )
    n_batches = ( data_size // batch_size)
    
    for epoch in range(epoch_size):
        for i in range(n_batches):

            generator_A.zero_grad()
            generator_B.zero_grad()
            discriminator_A.zero_grad()
            discriminator_B.zero_grad()
            discriminator_S.zero_grad()

            A = Variable(next(iter(A_loader)))
            B = Variable(next(iter(A_loader)))
            
            if cuda:
                A = A.cuda()
                B = B.cuda()
            
            #A = A.unsqueeze(1)
            AB, AL_feats, LAB_feats = generator_B(A)
            ABA, ABL_feats, ABLA_feats = generator_A(AB)
            #B = B.unsqueeze(1)
            BA, BL_feats, LBA_feats = generator_A(B)
            BAB, BAL_feats, BALB_feats = generator_B(BA)
            
            recon_loss_BA = recon_criterion( BA, B)
            recon_loss_AB = recon_criterion( AB, A)
            recon_loss_ABA = recon_criterion( ABA, A)
            recon_loss_BAB = recon_criterion( BAB, B)
            
            print('recon: ', recon_loss_BA)
            
            break

        break
    #for index, imgs in enumerate(dataloader):
    #    print(index, imgs.shape) 
        
        
    #    data  = Variable(imgs)
    #    print(generator_A(data))
    #    break
    
    #print(generator_A(data))

    #test_A_V = Variable( torch.FloatTensor( test_A ), volatile=True)









