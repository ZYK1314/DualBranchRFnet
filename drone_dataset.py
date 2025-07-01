import random
import numpy as np
import pandas as pd
import copy
import os
import time
from datetime import datetime
import torch
import matplotlib.pyplot as plt
from scipy import signal
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import Spectrogram
from torchvision.transforms import Compose, Resize, ToTensor
from sklearn.model_selection import train_test_split

from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from tqdm import tqdm  # progressbar
import torchmetrics

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import gc


class drone_data_dataset(Dataset):
    """
    Dataset class for drone IQ Signals + transform to spectrogram
    """
    def __init__(self, path, transform=None, device=None):
        self.path = path
        self.files = os.listdir(path)
        self.files = [f for f in self.files if f.endswith('pt')] # filter for files with .pt extension  
        self.files = [f for f in self.files if f.startswith('IQdata_sample')] # filter for files which start with IQdata_sample in name
        self.transform = transform
        self.device = device

        # create list of tragets and snrs for all samples
        self.targets = []
        self.snrs = []
        
        for file in self.files:
            self.targets.append(int(file.split('_')[2][6:])) # get target from file name
            self.snrs.append(int(file.split('_')[3].split('.')[0][3:])) # get snr from file name

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        sample_id = int(file.split('_')[1][6:]) # get sample id from file name
        data_dict = torch.load(self.path + file, weights_only=True) # load data       
        iq_data = data_dict['x_iq']
        act_target = data_dict['y']
        act_snr = data_dict['snr']

        if self.transform:
            if self.device:
                iq_data = iq_data.to(device=device)
            transformed_data = self.transform(iq_data)
        else:
            transformed_data = None

        return iq_data, act_target, act_snr, sample_id, transformed_data
    
    def get_targets(self): # return list of targets
        return self.targets

    def get_snrs(self): # return list of snrs
        return self.snrs
    
    def get_files(self):
        return self.files


class transform_spectrogram(torch.nn.Module):
    def __init__(
        self,
        device,
        n_fft=1024,
        win_length=1024,
        hop_length=1024,
        window_fn=torch.hann_window,
        power=None, # Exponent for the magnitude spectrogram, (must be > 0) e.g., 1 for magnitude, 2 for power, etc. If None, then the complex spectrum is returned instead. (Default: 2)
        normalized=False,
        center=False,
        #pad_mode='reflect',
        onesided=False
    ):
        super().__init__()
        self.spec = Spectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length, window_fn=window_fn, power=power, normalized=normalized, center=center, onesided=onesided).to(device=device)   
        self.win_lengt = win_length
        self.transform = Compose([Resize((self.win_lengt//2, self.win_lengt//2))])

    def forward(self, iq_signal: torch.Tensor) -> torch.Tensor:
        # Convert to spectrogram
        iq_signal = iq_signal[0,:] + (1j * iq_signal[1,:]) # convert to complex signal
        spec = self.spec(iq_signal)
        spec = torch.view_as_real(spec) # Returns a view of a complex input as a real tensor. last dimension of size 2 represents the real and imaginary components of complex numbers
        spec = torch.moveaxis(spec,2,0) # move channel dimension to first dimension (1024, 1024, 2) -> (2, 1024, 1024)
        spec = spec/self.win_lengt # normalise by fft window size
        spec = self.transform(spec)
        return spec


def plot_two_channel_spectrogram(spectrogram_2d, title='', figsize=(10,6)):
    figure, axis = plt.subplots(1, 2, figsize=figsize)
    re = axis[0].imshow(spectrogram_2d[0,:,:]) #, aspect='auto', origin='lower')
    axis[0].set_title("Re")
    figure.colorbar(re, ax=axis[0], location='right', shrink=0.5)

    im = axis[1].imshow(spectrogram_2d[1,:,:]) #, aspect='auto', origin='lower')
    axis[1].set_title("Im")
    figure.colorbar(im, ax=axis[1], location='right', shrink=0.5)

    figure.suptitle(title)
    plt.show()


def plot_two_channel_iq(iq_2d, title='', figsize=(10,6)):
    figure, axis = plt.subplots(2, 1, figsize=figsize)
    axis[0].plot(iq_2d[0,:]) 
    axis[0].set_title("Re")
    axis[1].plot(iq_2d[1,:])
    axis[1].set_title("Im")
    figure.suptitle(title)
    plt.show()

project_path = './'
result_path = project_path + 'results/experiments/'
# data_path = './data/'
data_path = './data/'

# global params
# num_workers = 0  # number of workers for data loader
num_folds =  5  # number of folds for cross validation
# num_epochs = 100 # number of epochs to train
# batch_size = 100 # batch size
# learning_rate = 0.01 # start learning rate
# train_verbose = True  # show epoch
# model_name = 'vgg16_bn'

# set device
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

def get_drone_data_dataset(batch_size):
    # mp.set_start_method('spawn')  # set multiprocessing context to 'spawn'
    print(torch.cuda.is_available())

    # 清理 GPU 缓存
    torch.cuda.empty_cache()
    gc.collect()
    
    # experiment_time = datetime.now().strftime('%m-%d_%H-%M')
    # experiment_name = 'time' + str(experiment_time) + '_' + \
    #                 model_name + \
    #                 '_CV' + str(num_folds) + \
    #                 '_epochs' + str(num_epochs) + \
    #                 '_lr' + str(learning_rate) + \
    #                 '_batchsize' + str(batch_size)


    # print('Starting experiment:\n', experiment_name)

    # create path to store results
    # act_result_path = result_path + experiment_name + '/'
    # try:
    #     os.mkdir(act_result_path)
    #     print("make dir " + str(act_result_path))
    # except OSError as error:
    #     print(error)

    # try:
    #     os.mkdir(act_result_path + 'plots/')
    #     print("make dir " + str(act_result_path + 'plots/'))
    # except OSError as error:
    #     print(error)

    random.seed(0)  # 保证随机结果可复现
    
    # read statistics/class count of the dataset
    dataset_stats = pd.read_csv(data_path + 'class_stats.csv', index_col=0)
    class_names = dataset_stats['class'].values

    # read SNR count of the dataset
    snr_stats = pd.read_csv(data_path + 'SNR_stats.csv', index_col=0)
    snr_list = snr_stats['SNR'].values

    # setup transform: IQ -> SPEC
    data_transform = transform_spectrogram(device=device, n_fft=1024, win_length=1024, hop_length=1024) # create transform object
    # create dataset object
    drone_dataset = drone_data_dataset(path=data_path + '/snr30/', device=device, transform=data_transform)
    print('Dataset size: ', len(drone_dataset))
    # split data with stratified kfold
    dataset_indices = list(range(len(drone_dataset)))

    # targets = drone_dataset.get_targets()
    # snr_list = drone_dataset.get_snrs()
    # files = drone_dataset.get_files()

    # split data with stratified kfold with respect to target class
    train_idx, test_idx = train_test_split(dataset_indices, test_size=1/num_folds, stratify=drone_dataset.get_targets())
    y_test = [drone_dataset.get_targets()[x] for x in test_idx]
    y_train = [drone_dataset.get_targets()[x] for x in train_idx]

    # split val data from train data in stratified k-fold manner
    train_idx, val_idx = train_test_split(train_idx, test_size=1/num_folds, stratify=y_train)
    y_val = [drone_dataset.get_targets()[x] for x in val_idx]
    y_train = [drone_dataset.get_targets()[x] for x in train_idx]

    # get train samples weight by class weight for each train target
    class_weights = 1. / dataset_stats['count']

    train_samples_weight = np.array([class_weights[int(i)] for i in y_train])
    train_samples_weight = torch.from_numpy(train_samples_weight)

    train_dataset = torch.utils.data.Subset(drone_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(drone_dataset, val_idx)
    test_dataset = torch.utils.data.Subset(drone_dataset, test_idx)

    # define weighted random sampler with the weighted train samples
    train_sampler = torch.utils.data.WeightedRandomSampler(train_samples_weight.type('torch.DoubleTensor'), len(train_samples_weight))

    batch_size = batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=nw,
        pin_memory=True)

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=nw,
        pin_memory=True)

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=nw,
        pin_memory=False,
        persistent_workers=False)

    dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset), 'test': len(test_dataset)}

    num_classes = len(np.unique(y_val))

    return dataloaders, num_classes, class_names

