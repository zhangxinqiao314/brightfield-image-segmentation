# General utilities:
import requests
from time import time
import datetime
import os
from os.path import join as pjoin
import glob as glob
from pdb import set_trace as bp
import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import h5py
import math
import pickle
from collections import OrderedDict
from tqdm import tqdm
from pprint import pprint

#Vis
import imageio
from sklearn.preprocessing import StandardScaler
from skimage.color import rgb2gray
from skimage.filters import difference_of_gaussians, window
from skimage import morphology
from skimage.filters import *
from skimage.transform import rescale
from skimage.feature import peak_local_max
from skimage import data, img_as_float
from scipy import ndimage
from scipy.fft import fftn, fftshift
import cv2
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

# NN
import torch
import torch.autograd as autograd
from torch.autograd import Variable
from torch.autograd import Function
import torchvision
from torchvision import datasets, models
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

# My Utilities
import utils

printing = {'PNG':True,
            'EPS':False, 
           'dpi': 300}
           

#######################################################################################

class conv_block(nn.Module):
    def __init__(self,t_size,n_step):
        super(conv_block,self).__init__()
        self.cov1d_1 = nn.Conv2d(t_size,t_size,3,stride=1,padding=1,padding_mode = 'zeros')
        self.cov1d_2 = nn.Conv2d(t_size,t_size,3,stride=1,padding=1,padding_mode = 'zeros')
        self.cov1d_3 = nn.Conv2d(t_size,t_size,3,stride=1,padding=1,padding_mode = 'zeros')
#         self.attention_1 = nn.MultiheadAttention(n_step, 1)
#         self.attention_2 = nn.MultiheadAttention(n_step, 1)
#         self.attention_3 = nn.MultiheadAttention(n_step, 1)
#         self.norm_1 = nn.LayerNorm([n_step])
#         self.norm_2 = nn.LayerNorm([n_step])
        self.norm_3 = nn.LayerNorm(n_step)
#        self.drop = nn.Dropout(p=0.2)
#         self.relu_1 = nn.ReLU()
#         self.relu_2 = nn.ReLU()
#         self.relu_3 = nn.ReLU()
        self.relu_4 = nn.ReLU()
        
    def forward(self,x):
        x_input = x
        out = self.cov1d_1(x)        
        out = self.cov1d_2(out)
        out = self.cov1d_3(out)
        out = self.norm_3(out)        
        out = self.relu_4(out)
        out = out.add(x_input)
#        output = self.drop(x)
        
        return out

#######################################################################################

class identity_block(nn.Module):
    def __init__(self,t_size,n_step):
        super(identity_block,self).__init__()
        self.cov1d_1 = nn.Conv2d(t_size,t_size,3,stride=1,padding=1,padding_mode = 'zeros')
#         self.attention_1 = nn.MultiheadAttention(n_step, 1)
        self.norm_1 = nn.LayerNorm(n_step)
#        self.drop = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
        
        
    def forward(self,x):
        x_input = x
        out = self.cov1d_1(x)
        out = self.norm_1(out)
        out = self.relu(out)

        
        return out

#######################################################################################

class Encoder(nn.Module):
    def __init__(self,original_step_size,pool_list,embedding_size,conv_size,device):
        super(Encoder,self).__init__()
        
        blocks = []
        self.input_size_0 = original_step_size[0]
        self.input_size_1 = original_step_size[1]
        number_of_blocks = len(pool_list)
        blocks.append(conv_block(t_size=conv_size, n_step=original_step_size))
        blocks.append(identity_block(t_size=conv_size, n_step=original_step_size))
        blocks.append(nn.MaxPool2d(pool_list[0], stride=pool_list[0]))
        
        for i in range(1,number_of_blocks):
            original_step_size = [original_step_size[0]//pool_list[i-1],original_step_size[1]//pool_list[i-1]]
            blocks.append(conv_block(t_size=conv_size, n_step=original_step_size))
            blocks.append(identity_block(t_size=conv_size, n_step=original_step_size))
            blocks.append(nn.MaxPool2d(pool_list[i], stride=pool_list[i])) 
            
        self.block_layer = nn.ModuleList(blocks)
        self.layers=len(blocks)
        original_step_size = [original_step_size[0]//pool_list[-1],original_step_size[1]//pool_list[-1]]
        
        input_size = original_step_size[0]*original_step_size[1]
        self.cov2d = nn.Conv2d(1,conv_size,3,stride=1,padding=1,padding_mode = 'zeros')
        self.cov2d_1 = nn.Conv2d(conv_size,1,3,stride=1,padding=1,padding_mode = 'zeros')

        self.relu_1 = nn.ReLU()

        self.dense1 = nn.Linear(input_size,embedding_size)
#         self.dense2 = nn.Linear(input_size,embedding_size)
        self.device=device

        
    def forward(self,x):
#        x = x.transpose(1,2)
        out = x.view(-1,1,self.input_size_0,self.input_size_1)
#        x = self.average(x)
        out = self.cov2d(out)
        for i in range(self.layers):
            out = self.block_layer[i](out)
        out = self.cov2d_1(out)
#         print(out.shape)
        out = torch.flatten(out,start_dim=1)
#         print(out.shape)
#        print(x.shape)
#         x = x.transpose(1,2)
#        encode,(_,__) = self.lstm(x)
#        encode = encode[:,-1,:]
        out = self.dense1(out)   
        selection = self.relu_1(out)
#         out_std = self.dense2(out)
        
        
        scale_1 = nn.Tanh()(out[:,0])*0.1+1
        scale_2 = nn.Tanh()(out[:,1])*0.1+1
        
        trans_1 = out[:,3]
        trans_2 = out[:,4]
        
        rotate = out[:,2]
        
        a_1 = torch.cos(rotate)
#       a_2 = -torch.sin(selection)
        a_2 = torch.sin(rotate)
        a_4 = torch.ones(rotate.shape).to(self.device)
        a_5 = rotate*0
        
        b1 = torch.stack((a_1,a_2), dim=1).squeeze()
        b2 = torch.stack((-a_2,a_1), dim=1).squeeze()
        b3 = torch.stack((a_5,a_5), dim=1).squeeze()
        rotation = torch.stack((b1, b2, b3),dim=2)
        
        c1 = torch.stack((scale_1,a_5), dim=1).squeeze()
        c2 = torch.stack((a_5,scale_2), dim=1).squeeze()
        c3 = torch.stack((a_5,a_5), dim=1).squeeze()
        scaler = torch.stack((c1, c2, c3),dim=2)

        d1 = torch.stack((a_4,a_5), dim=1).squeeze()
        d2 = torch.stack((a_5,a_4), dim=1).squeeze()
        d3 = torch.stack((trans_1,trans_2), dim=1).squeeze()
        translation = torch.stack((d1, d2, d3),dim=2)
        
        size_grid = torch.ones([x.shape[0],1,2,2])
        grid_1 = F.affine_grid(rotation.to(self.device), size_grid.size()).to(self.device)
        grid_2 = F.affine_grid(scaler.to(self.device), size_grid.size()).to(self.device)
        grid_3 = F.affine_grid(translation.to(self.device), size_grid.size()).to(self.device)

        final_out = torch.stack((selection, grid_1.reshape(x.shape[0],-1), grid_2.reshape(x.shape[0],-1), grid_3.reshape(x.shape[0],-1)), dim=1).squeeze()
        
        return final_out, rotation, scaler, translation

#######################################################################################

class Decoder(nn.Module):
    def __init__(self,original_step_size,up_list,pool_list,embedding_size,conv_size):
        super(Decoder,self).__init__() 
        self.input_size_0 = original_step_size[0]
        self.input_size_1 = original_step_size[1]
        self.dense = nn.Linear(embedding_size+8*3,original_step_size[0]*original_step_size[1])
        self.cov2d = nn.Conv2d(1,conv_size,3,stride=1,padding=1,padding_mode = 'zeros')
        self.cov2d_1 = nn.Conv2d(conv_size,1,3,stride=1,padding=1,padding_mode = 'zeros')
        
        blocks = []
        number_of_blocks = len(pool_list)
        blocks.append(conv_block(t_size=conv_size, n_step=original_step_size))
        blocks.append(identity_block(t_size=conv_size, n_step=original_step_size))
        
        for i in range(number_of_blocks):
            blocks.append(nn.Upsample(scale_factor=up_list[i], mode='bilinear', align_corners=True))
            original_step_size = [original_step_size[0]*up_list[i],original_step_size[1]*up_list[i]]
            blocks.append(conv_block(t_size=conv_size, n_step=original_step_size))
            blocks.append(identity_block(t_size=conv_size, n_step=original_step_size))
            
        self.block_layer = nn.ModuleList(blocks)
        self.layers=len(blocks)
        
        self.output_size_0 = original_step_size[0]
        self.output_size_1 = original_step_size[1]

        
    def forward(self,x):
        x=x.reshape(-1,embedding_size+8*3)
        out = self.dense(x)
        out = out.view(-1,1,self.input_size_0,self.input_size_1)
        out = self.cov2d(out)
        for i in range(self.layers):
            out = self.block_layer[i](out)
        out = self.cov2d_1(out)
        output = out.view(-1, self.output_size_0, self.output_size_1)
        
        return output
        

#######################################################################################

class VAE(nn.Module):
    ''' This the VAE, which takes a encoder and decoder.

    '''

    def __init__(self, enc, dec):
        super().__init__()

        self.enc = enc
        self.dec = dec

    def forward(self, x):
        # encode
        encoded = self.enc(x)

        # decode
        predicted = self.dec(encoded[0])
        
        return encoded, predicted

#######################################################################################

class Regularization(nn.Module):
    def __init__(self, model, weight_decay, p=2):
        '''
        :param model
        :param weight_decay: coeifficient of 
        :param p: p=1 is l1 regularization, p=2 is l2 regularizaiton
        '''
        super(Regularization, self).__init__()
        if weight_decay < 0:
            print("param weight_decay can not <0")
            exit(0)
        self.model = model
        self.weight_decay = weight_decay
        self.p = p
        self.weight_list = self.get_weight(model)

    #       self.weight_info(self.weight_list)

    def to(self, device):
        '''
        :param device: cude or cpu
        :return:
        '''
        self.device = device
        super().to(device)
        return self

    def forward(self, model):
        self.weight_list = self.get_weight(model)  
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss

    def get_weight(self, model):
        '''
        :param model: model
        :return: list of layers needs to be regularized  
        '''
        weight_list = []
        for name, param in model.named_parameters():
            if 'dec' in name and 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def regularization_loss(self, weight_list, weight_decay, p):
        '''
        :param weight_list: list of layers needs to be regularized  
        :param p: p=1 is l1 regularization, p=2 is l2 regularizaiton
        :param weight_decay: coeifficient
        :return: loss
        '''
        # weight_decay=Variable(torch.FloatTensor([weight_decay]).to(self.device),requires_grad=True)
        # reg_loss=Variable(torch.FloatTensor([0.]).to(self.device),requires_grad=True)
        # weight_decay=torch.FloatTensor([weight_decay]).to(self.device)
        # reg_loss=torch.FloatTensor([0.]).to(self.device)
        reg_loss = 0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg

        reg_loss = weight_decay * reg_loss
        return reg_loss

    def weight_info(self, weight_list):
        '''
        :param weight_list:
        :return: list of layers' name needs to be regularized  
        '''
        print("---------------regularization weight---------------")
        for name, w in weight_list:
            print(name)
            
#######################################################################################

def loss_function(model,
                  encoder,
                  decoder,
                  train_iterator,
                  optimizer,
                  device,
                  coef = 0, 
                  coef1 = 0,
                  ln_parm = 1, 
                  beta = None):

    weight_decay = coef
    weight_decay_1 = coef1
#     print(train_iterator)
    # set the train mode
    model.train()

    # loss of the epoch
    train_loss = 0
#     print(train_iterator)
    #    for i, x in enumerate(train_iterator):
    for x in tqdm(train_iterator, leave=True, total=len(train_iterator)):
        
        reg_loss_2 = Regularization(model, weight_decay_1, p=2).to(device)
     
        x_ = x.to(device, dtype=torch.float)

        # update the gradients to zero
        optimizer.zero_grad()

        if beta is None: 
          embedding = encoder(x_)[0]
        
        else:
          # forward pass
          #        predicted_x = model(x)
          embedding,sd,mn = encoder(x_)
        
        if weight_decay > 0:
            reg_loss_1 = weight_decay * torch.norm(embedding, ln_parm).to(device)
        else:
            reg_loss_1 = 0.0 
        
        encoded,predicted = model(x_)
        embedding = encoded[0]
        # reconstruction loss
#         print(x_.shape)
#         print(predicted.shape)
        loss = F.mse_loss(x_, predicted, reduction='mean')

        
        loss = loss +  reg_loss_1 #+reg_loss_2(model) 
            
        if beta is not None:
            vae_loss = beta * 0.5 * torch.sum(torch.exp(sd) + (mn)**2 - 1.0 - sd).to(device)
            vae_loss/= (sd.shape[0]*sd.shape[1])
        else:
            vae_loss=0

        loss = loss + vae_loss
        
        # backward pass
        train_loss += loss.item()

        loss.backward()
        # update the weights
        optimizer.step()

    return train_loss

#######################################################################################

def Train(model,encoder,decoder,train_iterator,optimizer,
          epochs,coef=0,coef_1=0,ln_parm=1, beta=None, epoch_ = None,folder='BF_jVAE'):
#     N_EPOCHS = epochs
    best_train_loss = float('inf')
    epoch = epoch_
    
    today = datetime.datetime.now()
    date = today.strftime('(%Y-%m-%d, %H:%M)')
            
#     if epoch_==None:
#         start_epoch = 0
#     else:
#         start_epoch = epoch_+1

#     print('train ',train_iterator)
#     for epoch in range(start_epoch,N_EPOCHS):
    train = loss_function(model,encoder,decoder,train_iterator,
                          optimizer,coef,coef_1,ln_parm,beta)
    train_loss = train
    train_loss /= len(train_iterator)
#        VAE_L /= len(train_iterator)
    print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, coef: {coef:.7f}')
#        print(f'......... VAE Loss: {VAE_L:.4f}')
#     print('.............................')
  #  schedular.step()
    if best_train_loss > train_loss:
        best_train_loss = train_loss
        patience_counter = 1
        checkpoint = {
            "net": model.state_dict(),
            "encoder":encoder.state_dict(),
            "decoder":decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            "epoch": epoch,
        }


        if epoch >=0:
            torch.save(checkpoint,
                       f'./{folder}/{date}_epoch:{epoch:05d}_trainloss:{train_loss:.4f}_coef:{coef:.4E}.pkl')