from __future__ import division
import os
import time
import glob
import datetime
import argparse
import numpy as np

import cv2
from PIL import Image
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

parser = argparse.ArgumentParser()
parser.add_argument("--noisetype", type=str, default="gauss25")
parser.add_argument('--data_dir', type=str, default='./Imagenet_val')
parser.add_argument('--val_dirs', type=str, default='./validation')
parser.add_argument('--save_model_path', type=str, default='./results')
parser.add_argument('--log_name', type=str, default='unet_gauss25_b4e100r02')
parser.add_argument('--gpu_devices', default='0', type=str)
parser.add_argument('--parallel', action='store_true')
parser.add_argument('--n_feature', type=int, default=48)
parser.add_argument('--n_channel', type=int, default=3)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--n_snapshot', type=int, default=1)
parser.add_argument('--batchsize', type=int, default=4)
parser.add_argument('--patchsize', type=int, default=256)
parser.add_argument("--Lambda1", type=float, default=1.0)
parser.add_argument("--Lambda2", type=float, default=1.0)
parser.add_argument("--increase_ratio", type=float, default=2.0)

opt, _ = parser.parse_known_args()
systime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
operation_seed_counter = 0

def checkpoint(net, epoch, name):
    save_model_path = os.path.join(opt.save_model_path, opt.log_name, systime)
    os.makedirs(save_model_path, exist_ok=True)
    model_name = 'epoch_{}_{:03d}.pth'.format(name, epoch)
    save_model_path = os.path.join(save_model_path, model_name)
    torch.save(net.state_dict(), save_model_path)
    print('Checkpoint saved to {}'.format(save_model_path))


def get_generator():
    global operation_seed_counter
    operation_seed_counter += 1
    g_cuda_generator = torch.Generator(device="cuda")
    g_cuda_generator.manual_seed(operation_seed_counter)
    return g_cuda_generator


class AugmentNoise(object):
    def __init__(self, style):
        #print(style)
        if style.startswith('gauss'):
            self.params = [
                float(p) / 255.0 for p in style.replace('gauss', '').split('_')
            ]
            if len(self.params) == 1:
                self.style = "gauss_fix"
            elif len(self.params) == 2:
                self.style = "gauss_range"
        elif style.startswith('poisson'):
            self.params = [
                float(p) for p in style.replace('poisson', '').split('_')
            ]
            if len(self.params) == 1:
                self.style = "poisson_fix"
            elif len(self.params) == 2:
                self.style = "poisson_range"

    def add_train_noise(self, x):
        shape = x.shape
        if self.style == "gauss_fix":
            std = self.params[0]
            std = std * torch.ones((shape[0], 1, 1, 1), device=x.device)
            noise = torch.cuda.FloatTensor(shape, device=x.device)
            torch.normal(mean=0.0,
                         std=std,
                         generator=get_generator(),
                         out=noise)
            return x + noise
        elif self.style == "gauss_range":
            min_std, max_std = self.params
            std = torch.rand(size=(shape[0], 1, 1, 1),
                             device=x.device) * (max_std - min_std) + min_std
            noise = torch.cuda.FloatTensor(shape, device=x.device)
            torch.normal(mean=0, std=std, generator=get_generator(), out=noise)
            return x + noise
        elif self.style == "poisson_fix":
            lam = self.params[0]
            lam = lam * torch.ones((shape[0], 1, 1, 1), device=x.device)
            noised = torch.poisson(lam * x, generator=get_generator()) / lam
            return noised
        elif self.style == "poisson_range":
            min_lam, max_lam = self.params
            lam = torch.rand(size=(shape[0], 1, 1, 1),
                             device=x.device) * (max_lam - min_lam) + min_lam
            noised = torch.poisson(lam * x, generator=get_generator()) / lam
            return noised

    def add_valid_noise(self, x):
        shape = x.shape
        if self.style == "gauss_fix":
            std = self.params[0]
            return np.array(x + np.random.normal(size=shape) * std,
                            dtype=np.float32)
        elif self.style == "gauss_range":
            min_std, max_std = self.params
            std = np.random.uniform(low=min_std, high=max_std, size=(1, 1, 1))
            return np.array(x + np.random.normal(size=shape) * std,
                            dtype=np.float32)
        elif self.style == "poisson_fix":
            lam = self.params[0]
            return np.array(np.random.poisson(lam * x) / lam, dtype=np.float32)
        elif self.style == "poisson_range":
            min_lam, max_lam = self.params
            lam = np.random.uniform(low=min_lam, high=max_lam, size=(1, 1, 1))
            return np.array(np.random.poisson(lam * x) / lam, dtype=np.float32)