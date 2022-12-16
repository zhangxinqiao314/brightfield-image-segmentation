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
import io
from pdb import set_trace as bp

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
from skimage.measure import profile_line
from skimage.transform import resize
from scipy import ndimage
from scipy.fft import fftn, fftshift
from scipy.signal import find_peaks
from scipy.ndimage import rotate
import cv2
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import PIL

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker

from scipy.signal import find_peaks
from scipy.ndimage import rotate
from skimage.measure import profile_line
from scipy.signal import savgol_filter
from scipy.fft import fft, ifft,fftfreq

# Pycroscopy
import sidpy
import pyNSID
print('sidpy version: ', sidpy.__version__)
import pycroscopy as px
from pycroscopy.image import ImageWindowing
import dask.array as da

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
import io_utils


printing = {'PNG':True,
            'EPS':False, 
           'dpi': 300}
           

#######################################################################################

def plot_window(x,y,image,windows):
    '''
    plot the filtered image, transform, and window
    x: x index
    y: y index
    image: image you generated windows from
    windows: dataset from h5 file generated from plotting windows
    '''
    
    
    fig = plt.figure(figsize=(16,8))
    gs = fig.add_gridspec(2, 3)
    axs = []
    axs.append( fig.add_subplot(gs[:,0:2]) ) # large subplot (2 rows, 2 columns)
    axs.append( fig.add_subplot(gs[0,2]) )   # small subplot (1st row, 3rd column)
    axs.append(fig.add_subplot(gs[1,2]))
    xi,yi = int((x+.5)*32),int((y+.5)*32)

    axs[0].set_title('Full Image')
    a1 = axs[0].imshow(image[:])
    rect = patches.Rectangle((xi+32,yi+32),64,64,linewidth=2, edgecolor='r', facecolor='none')
    axs[0].add_patch(rect)
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes('right', size='5%')
    cb=fig.colorbar(a1, cax=cax, orientation='vertical', pad = 0.2)
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')

    fft = windows[x][y]
    axs[1].set_title('fft tile')
    a2 = axs[1].imshow((np.log(fft)))
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes('right', size='5%')
    cb=fig.colorbar(a2, cax=cax, orientation='vertical', pad = 0.2)
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')

    # fft = dask.array.rechunk(fft,-1)
    # ifft = np.fft.ifft2(fft).real
    axs[2].set_title('unfiltered')
    a3 = axs[2].imshow(image[xi:xi+64,yi:yi+64],vmax=1.,vmin=0.)
    divider = make_axes_locatable(axs[2])
    cax = divider.append_axes('right', size='5%')
    cb=fig.colorbar(a3, cax=cax, orientation='vertical', pad = 0.2)
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')

    plt.tight_layout()
    plt.show()
    
##############################################################################################

def NormalizeData(data):
    if np.max(data)==np.min(data): return data
    else: return (data - np.min(data)) / (np.max(data) - np.min(data))

def fft_transform(image):
    '''
    assert r.dtype==torch.float32, 'Suppose to be torch.float32, not '+str(r.dtype)
    '''  
    
    image = torch.tensor(image)
    out = torch.clone(image)
    img_fft = torch.fft.fft2(image, dim=(0,1))
    img_shift = torch.fft.fftshift(img_fft)
    out = np.log(np.abs(img_shift))
    
    # abnormal value
    out[out==-np.inf] = 0
    
    # scale to 0-1
    out =  (out * 1/out.max())
    return out.numpy()
    
#############################################################################################

def temp(e):
    '''
    Gets temperature from file in format temp.png and determins ramp up or down
    '''
    if len(e)<8:
        return '-'+e[:-4].zfill(3)
    else:
        return '-0.png'

def get_temps(env):
    '''
    gets list of original images in order from folder with format: 
    ./environment_name/Ramp_(Up or Down)/temperature.png
    '''
    up = os.listdir(f'./{env}/Ramp_Up')
    up.sort(key=temp)
    up = list(map(lambda x:f'./{env}/Ramp_Up/'+x, up))

    down = os.listdir(f'./{env}/Ramp_Down')
    down.sort(key=temp,reverse=True)
    # down = down.map(function(val) {return '.'+val;})
    down = list(map(lambda x:f'./{env}/Ramp_Down/'+x, down))

    temps = up+down
    return temps
    
#############################################################################################
    
def write_cropped_filtered_h5(c1,c2,step,combined,temps):
  '''
  Crop and filter raw images and insert them into h5 file. Overwrites any exisiting crops
  c1: starting pixel
  c2: ending pixel
  step: pixel size to crop to
  combined: combined file name to write to
  temps: list of paths to BF images
  '''

  h = h5py.File(combined,'a') #put the name of the file you would like to use
  t_len=len(temps)

  if 'All' in h.keys(): 
    del h['All']
    h_write = h.create_dataset('All',(t_len,step,step),dtype='f4')
  h_write=h['All']

  if  'All_filtered' in h.keys(): 
    del h['All_filtered']
    h_writef = h.create_dataset('All_filtered',(t_len,step,step),dtype='f4')
  h_writef=h['All_filtered']

  for i,temp in enumerate(tqdm(temps)):
      im = rgb2gray(imageio.imread(temp))
  #     im[1850:,:400]=0 ## get rid of the scale
      im1 = im[c1:c1+step,c2:c2+step] # crop
      
      h_write[i] = im1
      
      img_blur = ndimage.gaussian_filter(im1,20)
      im1=im1-img_blur
      im1=NormalizeData(im1)

      h_writef[i] = im1
      
  h.close()
  
#############################################################################################

def write_fft_windows(images,combined,temps,parms_dict,
                      grp='fft_windows',dset='fft_windows_dataset',logset='fft_windows_logdata',
                      resize_size=128):
  '''
  Write fft windows into the combined h5 file.
  for more information on ImageWindowing, consult: 
     https://github.com/pycroscopy/pycroscopy/blob/main/pycroscopy/image/image_window.py

  images: h5 dataset or nparray of cropped/filtered BF images
  combined: (string) name of combined file
  temps: list of paths to image files
  parms_dict: list of parameters for ImageWindowing object to be created
  grp: name of group that contains windowed datasets
  dset: raw fft windowed images
  logset: log and rescaled dset
  '''
  h = h5py.File(combined,'a')
  images = images[:]
  iw = ImageWindowing(parms_dict)
  t_len = len(temps)
  
  if grp not in h.keys(): h_windows=h.create_group(grp)
  h_windows=h[grp]

  for i,image in enumerate(tqdm(images)):
      data_image = sidpy.Dataset.from_array(image)
      windows = iw.MakeWindows(data_image)
      title = re.split('/|\.',temps[i])[-3]+'_'+re.split('/|\.',temps[i])[-2]
      
      if f'filler' in h[grp]: del h[grp][f'filler']
      meas_grp = h[grp].create_group(f'filler')
      
      pyNSID.hdf_io.write_nsid_dataset(windows, meas_grp, main_data_name="windows");    
      a,b,x,y = h[grp]['filler']['windows']['windows'].shape
      data = h[grp]['filler']['windows']['windows'][:].reshape(-1,x,y)
      
      if dset not in h[grp].keys(): d_windows=h[grp].create_dataset(dset,shape=(t_len*a*b,x,y))
      d_windows=h[grp][dset]
      d_windows[i*a*b:(i+1)*a*b] = data

      if logset not in h[grp].keys(): logdata= h[grp].create_dataset(logset,shape=(t_len*a*b,1,resize_size,resize_size),dtype='f4')
      logdata=h[grp][logset]
      data = data.reshape(-1,1,x,y)
      data = resize(data,(a*b,1,resize_size,resize_size))
      data = np.log(data+1)
      data[data>5]=5
      logdata[i*a*b:(i+1)*a*b] = data
      
  #     print(h[grp].keys())

  h.close()
  
#############################################################################################

def make_folder(folder, **kwargs):
    """
    Function that makes new folders
    Parameters
    ----------'
    folder : string
        folder where to save
    Returns
    -------
    folder : string
        folder where to save
    """

    # Makes folder
    os.makedirs(folder, exist_ok=True)

    return (folder)

def find_nearest(array, value, averaging_number):
    """
    returns the indices nearest to a value in an image
    Parameters
    ----------
    array : float, array
        image to find the index closest to a value
    value : float
        value to find points near
    averaging_number : int
        number of points to find
    """
    idx = (np.abs(array-value)).argsort()[0:averaging_number]
    return idx

#############################################################################################

def savefig(filename, printing):

    """
    function that saves the figure

    :param filename: path to save file
    :type filename: string
    :param printing: contains information for printing
                     'dpi': int
                            resolution of exported image
                      print_EPS : bool
                            selects if export the EPS
                      print_PNG : bool
                            selects if print the PNG
    :type printing: dictionary

    """


    # Saves figures at EPS
    if printing['EPS']:
        plt.savefig(filename + '.eps', format='eps',
                    dpi=printing['dpi']#, bbox_inches='tight'
                   )

    # Saves figures as PNG
    if printing['PNG']:
        plt.savefig(filename + '.png', format='png',
                    dpi=printing['dpi'],#, bbox_inches='tight'
                    facecolor = 'white'
                   )

#############################################################################################

def make_movie(movie_name, input_folder, output_folder, file_format,
                            fps, output_format = 'mp4', reverse = False):

    """
    Function which makes movies from an image series

    Parameters
    ----------
    movie_name : string
        name of the movie
    input_folder  : string
        folder where the image series is located
    output_folder  : string
        folder where the movie will be saved
    file_format  : string
        sets the format of the files to import
    fps  : numpy, int
        frames per second
    output_format  : string, optional
        sets the format for the output file
        supported types .mp4 and gif
        animated gif create large files
    reverse : bool, optional
        sets if the movie will be one way of there and back
    """

    # searches the folder and finds the files
    file_list = glob.glob('./' + input_folder + '/*.' + file_format)

    # Sorts the files by number makes 2 lists to go forward and back
    list.sort(file_list)
    file_list_rev = glob.glob('./' + input_folder + '/*.' + file_format)
    list.sort(file_list_rev,reverse=True)

    # combines the file list if including the reverse
    if reverse:
        new_list = file_list + file_list_rev
    else:
        new_list = file_list


    if output_format == 'gif':
        # makes an animated gif from the images
        clip = ImageSequenceClip(new_list, fps=fps)
        clip.write_gif(output_folder + '/{}.gif'.format(movie_name), fps=fps)
    else:
        # makes and mp4 from the images
        clip = ImageSequenceClip(new_list, fps=fps)
        clip.write_videofile(output_folder + '/{}.mp4'.format(movie_name), fps=fps)

#############################################################################################

def layout_fig(graph, mod=None,size=3):
    """
    Sets the layout of graphs in matplotlib in a pretty way based on the number of plots
    Parameters
    ----------
    graphs : int
        number of axes to make
    mod : int (optional)
        sets the number of figures per row
    Returns
    -------
    fig : matplotlib figure
        handel to figure being created.
    axes : numpy array (axes)
        numpy array of axes that are created.
    """
#     print(graph)

    if mod is None:
        # Selects the number of columns to have in the graph
        if graph < 3:
            mod = 2
        elif graph < 5:
            mod = 3
        elif graph < 10:
            mod = 4
        elif graph < 17:
            mod = 5
        elif graph < 26:
            mod = 6
        elif graph < 37:
            mod = 7
        else:
            mod = graph
            
            
    mod =int(mod)
    graph = int(graph)

    # builds the figure based on the number of graphs and selected number of columns
    fig, axes = plt.subplots(graph // mod + (graph % mod > 0), mod,
                             figsize=(size * mod, size * (graph // mod + (graph % mod > 0))))

    axes = axes.reshape(-1)
    for i, ax in enumerate(axes):
        if i >= graph: 
            fig.delaxes(ax)
    
    # deletes extra unneeded axes
    
    axes = axes[:graph]

    return (fig, axes)
    
#############################################################################################

def add_colorbar(fig,ax,sf=2):
    '''
    fig: Figure objexct
    ax: a = ax.imshow() ImageAxis object
    '''
    
    a=ax.axes
    divider = make_axes_locatable(a)
    cax = divider.append_axes('right', size='5%',pad='3%')
    cb=fig.colorbar(ax, cax=cax, orientation='vertical',format = f'%1.{sf}f')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')
    tick_locator = ticker.LinearLocator(5)
    cb.locator = tick_locator
    cb.update_ticks()
    fig.tight_layout()
    # plt.show()

#############################################################################################

def calculate_embeddings(logdata,emb,h_dataset_name,h_transforms_name,encoder,
                          embedding_size=8,batch_size=16,device=torch.device('cuda:0')):
  '''
  logdata: nparray or h5 file, (t*a*b,1,x,y) shaped log data from combined file.
  emb: (string) Name of embedding File
  h_dataset_name,h_transforms_name, Date+epoch+dataset/transforms
  encoder: 
                          embedding_size=8,batch_size=16,device=torch.device('cuda:0')):
  '''

  train_iterator = DataLoader(logdata, batch_size=batch_size,
                          shuffle=False)
  
  embedding_ = np.zeros([logdata.shape[0],embedding_size])
  rotation_ = np.zeros([logdata.shape[0],6])
  translation_ = np.zeros([logdata.shape[0],6])
  scaling_ = np.zeros([logdata.shape[0],6])

  h = h5py.File(emb,'a')

  j = 0
  for i, x in enumerate(tqdm(train_iterator)):
      if h_dataset_name not in h.keys(): h_write_e = h.create_dataset(h_dataset_name,(logdata.shape[0],4,embedding_size),dtype='f4')
      h_write_e=h[h_dataset_name]
      
      if h_transforms_name not in h.keys(): h_write_t = h.create_dataset(h_transforms_name,(logdata.shape[0],3,6),dtype='f4')   
      h_write_t=h[h_transforms_name]
        
      with torch.no_grad():
          value = x
          test_value = Variable(value.to(device))
          test_value = test_value.float()
          encoded = encoder(test_value)
          b_len = x.shape[0]
          
          embedding = encoded[0].squeeze().to('cpu').detach().numpy()
          h_write_e[i*b_len:(i+1)*b_len] = embedding
      
          rotation = encoded[1].reshape(b_len,6).to('cpu').detach().numpy()
          scaling = encoded[2].reshape(b_len,6).to('cpu').detach().numpy()
          translation = encoded[3].reshape(b_len,6).to('cpu').detach().numpy()
          h_write_t[i*b_len:(i+1)*b_len] = np.stack([rotation,translation,scaling],axis=1)
          
          j=j+1
  print(j)
  h.close()
  
#############################################################################################
  
def real_space_affine(rotation_1,translation_1,scaling_1,
                  t_len,a,b):
    '''
    Calculate the rotation in degrees, absoluate value of scaling, and absolute value of translation
    rotation_1,translation_1,scaling_1: Affine elements of embedding reshaped to (t_len,a,b)
    t_len: number of temperatures
    a,b: number of windows generated along each side of original BF image
    '''
    
    xyscaling = np.zeros((t_len,a,b))
    rotations = np.zeros((t_len,a,b))
    translations = np.zeros((t_len,a,b))
    
    for t in range(t_len):
      for i in range(a):
          for j in range(b):
              
              acos = np.arccos(rotation_1[t,i,j,0])
              asin = np.arcsin(rotation_1[t,i,j,3])
    
              if acos>0:
                  if asin<0: theta = 2*np.pi+asin
                  else: theta = asin
              if acos<0:
                  if asin<0: theta = theta = np.pi-asin
                  else: theta = acos
    
              rotations[t,i,j] = theta
              xyscaling[t,i,j] = np.sqrt(np.square(scaling_1[t,i,j,0]) + np.square(scaling_1[t,i,j,4]))
              translations[t,i,j] = np.sqrt(np.square(translation_1[t,i,j,2]) + np.square(translation_1[t,i,j,5]))
    return xyscaling, rotations, translations 

#############################################################################################
  
def layout_embedding_images(embedding_1,rotation_1,translation_1,scaling_1,
                            t_len,a,b,
                            combined,f,date,temps):

    xyscaling, rotations, translations = real_space_affine(rotation_1,translation_1,scaling_1,t_len,a,b)
    max_e,min_e = embedding_1[:8].max(),0
#   max_s,min_s = xyscaling.max(), scaling_1.min()
    max_t,min_t = translation_1.max(), translation_1.min()
    max_r,min_r = math.pi,-math.pi
    folder = make_folder(f'./{f}/{date} combined/')
    hf = h5py.File(combined,'r')
    embedding_size = embedding_1.shape[3]/4


    # make images of embeddings+original image
    for t,temp in enumerate(tqdm(temps)):
        title = re.split('/|\.',temp)[-3]+' '+re.split('/|\.',temp)[-2]+'$^{\circ}$C'
        orig = hf['All'][t]

        plt.ioff()
        fig = plt.figure(figsize=(18,12));
        gs = fig.add_gridspec(4,6);
        axs = []
        axs.append( fig.add_subplot(gs[:,:2]) ) # large subplot (2 rows, 2 columns)
        axs.append( fig.add_subplot(gs[:,2:4]) ) # large subplot (2 rows, 2 columns)
        axs.append( fig.add_subplot(gs[:,4:6]) ) # large subplot (2 rows, 2 columns)
        fig.suptitle(title);
        axs[0].set_title('Full Image');
        ax = axs[0].imshow(orig);
        add_colorbar(fig,ax);

        #Embeddings

        axs[1].set_title('Embeddings');
        axs[1].axis('off');
        fig_e, axes_e = layout_fig(embedding_size, mod=2);
        for j, axe in enumerate(axes_e):
            ax=axe.imshow(embedding_1[t,:,:,j].T,vmin=min_e,vmax=max_e);
            axe.set_title(f'emb {j}')
            add_colorbar(fig_e,ax);
        fig_e.tight_layout();
        img_buf = io.BytesIO();
        fig_e.savefig(img_buf,bbox_inches='tight',format='png');
        im = PIL.Image.open(img_buf);
        axs[1].imshow(im);
        img_buf.close()


        #Transforms

        axs[2].set_title('Transforms');
        axs[2].axis('off');
        fig_t, axes = layout_fig(6, mod=2);
        title = re.split('/|\.',temp)[-3]+'_'+re.split('/|\.',temp)[-2]

        ax=axes[0].imshow(scaling_1[t,:,:,0].T);
        axes[0].set_title('x scaling');
        add_colorbar(fig,ax);

        ax=axes[1].imshow(scaling_1[t,:,:,4].T);
        axes[1].set_title('y scaling');
        add_colorbar(fig,ax);

        ax=axes[2].imshow(xyscaling[t].T,);
        axes[2].set_title('combined scaling');
        add_colorbar(fig,ax);

        ax=axes[3].imshow(rotations[t].T,vmax=math.pi,vmin=-math.pi,cmap='twilight');
        axes[3].set_title('rotation (rad)');
        add_colorbar(fig,ax);

        ax=axes[4].imshow(translation_1[t,:,:,2].T,vmin=min_t,vmax=max_t);
        axes[4].set_title('x translation')
        add_colorbar(fig,ax)

        ax=axes[5].imshow(translation_1[t,:,:,5].T,vmin=min_t,vmax=max_t);
        axes[5].set_title('y translation');
        add_colorbar(fig,ax);

        fig_t.tight_layout();
        img_buf = io.BytesIO();
        fig_t.savefig(img_buf,bbox_inches='tight',format='png');
        im = PIL.Image.open(img_buf);
        axs[2].imshow(im);
        img_buf.close()


        fig.tight_layout();
        fig.savefig(folder+f'{t:02d}.png',facecolor='white');
        #   files.download(folder+f'{t:02d}.png');

#############################################################################################

class embedding_processing():
    def __init__(self,env,emb,combined,emb_checkpoint):
        '''
        env: (string) name of environment folder (Oxygen, Annealed,etc.)
        emb: (string) name of embedding h5 file
        combined: (string) name of combined h5 file
        emb_checkpoint: (string) name of embedding checkpoint. Titled in embedding h5 file
        '''
        self.temps = get_temps(env)
        self.t_len = len(self.temps)  
        self.env = env
        self.temp_labels = [re.split('\.|/',temp)[-2] for temp in self.temps]
        self.temp_labels[-1]=self.temp_labels[0]

        hf = h5py.File(combined,'r')
        h = h5py.File(emb,'r')

        sh = h[emb_checkpoint][:].shape
        a,b = int((sh[0]/self.t_len)**0.5),int((sh[0]/self.t_len)**0.5)
        self.orig_images=hf['All']
        self.orig_filtered=hf['All_filtered']
        self.embedding = h[emb_checkpoint][:].reshape(self.t_len,a,b,-1)
    
    def ezmask(self,image,thresh):
        mask = image > thresh
        # mask = morphology.binary_closing(mask)
        mask = morphology.binary_opening(mask,morphology.disk(2))
        mask = morphology.binary_closing(mask,morphology.disk(2))
        return mask
    
    def div_except(self,temp_idx,emb_idx):
        indices = list(range(8))
        indices.pop(emb_idx)
        return self.embedding[temp_idx,:,:,indices].sum(axis=(0))/7
    
    def make_mask(self,temp_idx, emb_idx, divide_idx=None, 
                  plot=True, save_folder=None,err_std=0):
        '''
        makes figure with image, warp, histogram, and mask (if specified). Returns binary mask.

        temp_idx: temperature
        emb_idx: 2D embedding channel to make a mask of.
        divide_idx: divide target channel another embedding
        plot: whether to show plot
        save_folder: where to save the image
        mask_range: number of std of threshold to determine error range for mask
        ---
        returns: mask
        '''
        length = 4
        im = self.embedding[temp_idx,:,:,emb_idx]
        image=im
        if image.max==0:
            mask = image
            thresh=0
        else:
            if divide_idx!=None:
                if divide_idx=='All': 
                    div = self.div_except(temp_idx,emb_idx) 
                else: 
                    div = self.embedding[[temp_idx,int(self.t_len/2)],:,:,divide_idx].sum(axis=0) 
                    # div = self.embedding[temp_idx,:,:,divide_idx].sum(axis=0) 
                image = image/(div+1)
                image = image-div
                image[image<0] = 0
                length = 5
                
            if image.max()==0: # if its 0
                mask=image
                thresh=0
            else:
                thresh = threshold_otsu(self.embedding[:,:,:,emb_idx])
                mask = self.ezmask(image,thresh).astype(int)
                
        if err_std>0:
            if image.max()==0: 
                mask0,mask1 = mask,mask
            else:
                mask0 = self.ezmask(image,max([0,thresh-im.std()*err_std])).astype(int)
                mask1 = self.ezmask(image,thresh+im.std()*err_std).astype(int)

        if plot==True:
            ## make figure
            fig,axes = layout_fig(length, mod=2)

            fig.set_figheight(10)
            fig.set_figwidth(10)
            fig.suptitle(f'{self.env} at {self.temp_labels[temp_idx]}$^\circ$C')

            axes[0].set_title('Original Image')
            a1=axes[0].imshow(self.orig_images[temp_idx])
            divider = make_axes_locatable(axes[0])
            cax = divider.append_axes('right', size='5%')
            cb=fig.colorbar(a1, cax=cax, orientation='vertical', pad = 0.2)
            cb.ax.xaxis.set_ticks_position('top')
            cb.ax.xaxis.set_label_position('top')

            axes[1].set_title(f'Embedding channel {emb_idx}')
            a1=axes[1].imshow(im.T)
            divider = make_axes_locatable(axes[1])
            cax = divider.append_axes('right', size='5%')
            cb=fig.colorbar(a1, cax=cax, orientation='vertical', pad = 0.2)
            cb.ax.xaxis.set_ticks_position('top')
            cb.ax.xaxis.set_label_position('top')

            axes[2].set_title(f'Histogram for Embedding')
            axes[2].hist(image.flatten(),bins=50)
            axes[2].axvline(thresh, color='k', ls='--')
            if err_std>0:
                axes[2].axvline(thresh+im.std()*err_std,color='r',ls='--')
                axes[2].axvline(thresh-im.std()*err_std,color='r',ls='--')

            axes[3].set_title(f'Mask')
            if err_std>0: a1=axes[3].imshow((mask+mask0+mask1).T,vmin=0,vmax=3)
            else: a1=axes[3].imshow(mask.T,vmin=0,vmax=1)
            divider = make_axes_locatable(axes[3])
            cax = divider.append_axes('right', size='5%')
            cb=fig.colorbar(a1, cax=cax, orientation='vertical', pad = 0.2)
            cb.ax.xaxis.set_ticks_position('top')
            cb.ax.xaxis.set_label_position('top')

            if divide_idx!=None:
                axes[4].set_title('Cleaned')
                a1=axes[4].imshow(image.T)
                divider = make_axes_locatable(axes[4])
                cax = divider.append_axes('right', size='5%')
                cb=fig.colorbar(a1, cax=cax, orientation='vertical', pad = 0.2)
                cb.ax.xaxis.set_ticks_position('top')
                cb.ax.xaxis.set_label_position('top')

            plt.tight_layout()
            plt.show()

            if save_folder!=None:
                plt.savefig(save_folder)
                
        if err_std>0:
            if image.max()==0: return image,image,image
            return mask,mask0,mask1
        else:
            return mask
    
    #####################################################################################

    def graph_relative_area(self,channels=range(8),masked=False,clean_div=None,smoothing=None,legends=None,
                            plot=True,err_std=0,save_folder=None):
        '''
        Makes a graph of the average intensity of selected embeddings channels across temperature range.
        Returns dictionary of domain structure and smooth values

        channels: (list) default is all channels. Othewise, specify indices
        masked: (Bool) Whether to calculate average area with only mask or with original embedding intensities
        clean_div: (list) Channels that can be used to eliminate stray signal in selected embedding channels
        smoothing: (int) convolution (smoothing) factor. Must be odd for odd length dataset, and even for even length dataset.
        legends: (list) Domain labels
        '''
        rel_areas_emb = np.zeros((len(channels),self.t_len))
        rel_areas_err = np.zeros((len(channels),self.t_len,2))
        
        for t in tqdm(range(self.t_len)):
            for i,c in enumerate(channels):
                im=self.embedding[t,:,:,c]
                if im.max()==0:
                    mask = np.zeros(im.shape)
                else:   
                    if clean_div!=None: 
                        if clean_div=='All': div = self.div_except(t,c)
                        else: div = self.embedding[t,:,:,clean_div[i]]
                        im = im/(self.embedding[int(self.t_len/2),:,:,2]+div+1)
                        im = im/(div+1)

                    if clean_div==None: 
                        mask = self.make_mask(t,c,plot=False,err_std=err_std)
                    elif clean_div=='All': 
                        mask = self.make_mask(t,c,divide_idx='All',plot=False,err_std=err_std)
                    else: 
                        mask = self.make_mask(t,c,divide_idx=clean_div[i],plot=False,err_std=err_std)
                
                if masked:
                    if err_std==0: rel_areas_emb[i][t] = mask.mean()
                    else:
                        rel_areas_emb[i][t] = mask[0].mean()
                        rel_areas_err[i][t][0] = mask[1].mean()
                        rel_areas_err[i][t][1] = mask[2].mean()
                else:   
                    # rel_areas_emb[i][t] = im.mean()
                    if err_std==0: 
                        rel_areas_emb[i][t] = (im*mask).mean()
                    else:
                        rel_areas_emb[i][t] = (im*mask[0]).mean()
                        rel_areas_err[i][t][0] = (im*mask[1]).mean()
                        rel_areas_err[i][t][1] = (im*mask[2]).mean()
        smooth_list=[]
        smooth_err=[[],[]]
        for i,r in enumerate(rel_areas_emb):
            if smoothing!=None: 
                x=int((smoothing-1)/2)
                rel_area_smooth = np.convolve(np.pad(r,x,mode='edge'), 
                                              np.ones(smoothing)/smoothing,'valid' )
                rel_area_smooth_err0 = np.convolve(np.pad(rel_areas_err[i,:,0],x,mode='edge'), 
                                                   np.ones(smoothing)/smoothing,'valid')
                rel_area_smooth_err1 = np.convolve(np.pad(rel_areas_err[i,:,1],x,mode='edge') ,
                                                   np.ones(smoothing)/smoothing,'valid')
                # rel_area_smooth = resize(rel_area_smooth.reshape(1,-1),(1,self.t_len)).flatten()
                smooth_list.append(rel_area_smooth)
                smooth_err[0].append(rel_area_smooth_err0)
                smooth_err[1].append(rel_area_smooth_err1)
            else: 
                smooth_list.append(r)
                if err_std:
                    smooth_err[0].append(rel_areas_err[i])
                    smooth_err[1].append(rel_areas_err[i])

        if plot==True:
            matplotlib.rcParams.update({'font.size': 16})
            t_half=int(self.t_len/2)
            x=np.linspace(0,self.t_len-1,self.t_len)
            new_labels=[]
            wanted_labels = ['23','120',f'{self.temp_labels[t_half]}']
            for val in self.temp_labels:
                if val in wanted_labels: new_labels.append(val)
                else: new_labels.append('')
                
                
            plt.Figure(figsize=(4,4),dpi=400)   
            plt.xticks(x, new_labels)
            # plt.ylabel('Area Fraction')
            # plt.xlabel('Temperature ($^\circ$C)')
            # plt.text(t_half-int(t_half/4), 0.6, '+$\Delta$T')
            # plt.text(t_half+int(t_half/10), 0.6, '-$\Delta$T')
            plt.suptitle(f'Relative Areas of {self.env}')
            
            for i,r in enumerate(smooth_list): # fill error bars
                plt.plot(x,r,'-o',linewidth=3)
                if err_std>0: 
                    plt.fill_between(x,smooth_err[0][i],smooth_err[1][i],
                                     alpha=0.25,label='_nolegend_')
                plt.ylim(0,1)
                
            plt.axvline(int(t_half), color='k', ls='--')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            # if legends!=None: plt.legend(legends)

        if save_folder!=None: 
            folder=make_folder(save_folder)
            plt.savefig(save_folder+f'/relative_area_{self.env}.png',facecolor='white')

        # print('line2 changed')
        return dict(zip(legends,smooth_list)) 
    
    
    #############################################################################################



    def reject_outliers(self,data, m=3):
        return data[abs(data - data.mean()) < m*data.std()]

    plt.rcParams["figure.figsize"] = (8,8)

    def calculate_densities(self,rel_areas,channels,div_channels=None,
                            legends_dict = {'<100> Horizontal':0,
                                            '<110> Left':-45,
                                            '<100> Vertical':90,
                                            '<110> Right':45},
                            freq_threshold=500):

        '''
        Calculates the mean spacing between needles for domains, perpendicular to needle direction using fft transform.
        Examines all temperatures.
        Returns:  dictionary with keys=domain direction and entries=mean linear density for a given environment
        
        rel_areas: (dictionary) list of relative areas to make for filter out unreliable data points
        channels: (list) Embedding channels with oriented needles
        div_channels: (list) Embedding channels most opposite to orientation in channels for cleaning
        legends_dict: (dictionary) keys=structure and location, values=slope of needle (R-->L) 
        freq_threshold: filter out high frequency signals (noise)
        '''
        

        for i,c in enumerate(channels):
            e = channels[i]
            d = div_channels[i]
            leg = list(legends_dict.keys())[i]
            angle = legends_dict[leg]
            rel_area_mean=sum(rel_areas[leg])/len(rel_areas[leg])

            temp_density_list = []
            skipped=0

            print(leg)

            for t,temp in enumerate(tqdm(self.temps)):
                # if the relative area is too small, it is noisy
                if rel_areas[leg][t]<=rel_area_mean:
                    skipped=skipped+1
                    continue

                temp_spacing=[]
                cleaned_sum_ffts=np.zeros((100,))    
                sh = self.orig_filtered[t].shape

                # make mask and scale to shape of orig image
                mask1 = self.make_mask(t,e,divide_idx=d,plot=False)
                sh1 = mask1.shape
                mask = resize(mask1,sh).T
                masked_im=mask*self.orig_filtered.shape[t]

                # rotate perpendicular to the domain needle direction
                rotated = rotate(masked_im, angle=90-angle, reshape=True)
                sh0=rotated.shape
                
                start_samples = np.linspace(0,sh0[0]-1,sh1[0]).astype(int)

                for s in start_samples:
                    start = (s, 0) #Start of the profile line row=100, col=0
                    end = (s, sh0[0]-1) #End of the profile line row=100, col=last

                    profile1 = profile_line(rotated, start, end,mode='constant')
                    profile = savgol_filter(profile1,71,2)

                    fft_profile=fft(1-profile)
                    N=len(fft_profile)
                    T=np.arange(N)
                    freq=fftfreq(N)
                    cleaned=abs(fft_profile)
                    cleaned[freq>freq_threshold]=0
                    cleaned_sum_ffts=cleaned_sum_ffts+cleaned[:100]

                cleaned_sum_ffts=cleaned_sum_ffts/len(start_samples)
                hist0=plt.hist(cleaned_sum_ffts,bins=100); #return counts,bins
                plt.close()
                hist=hist0[0][1:-1],hist0[1][1:-1] #exclude beginning and end of histogram
                ind=np.argsort(hist[0])
                counts,bins = hist[0][ind],hist[1][ind]
    #                 temp_counts=temp_counts+counts
                if np.sum(counts)>0:#len(start_samples)/15: 
                    temp_density_list.append(np.sum(counts*bins)/np.sum(counts))
                    # print(np.sum(counts),',',temp_density_list[-1])


            if len(temp_density_list)<self.t_len/10:
                   densities_dict[leg] = 0
            else: 
                   densities_dict[leg] = sum(temp_density_list)/len(temp_density_list)
            # print(skipped)


        return densities_dict