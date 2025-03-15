import numpy as np
from scipy.io import loadmat, savemat
import os
import time
import logging
import glob
import cv2
from PIL import Image
import torch
from torch.utils import data as data
from basicsr.utils.registry import DATASET_REGISTRY

def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size**2, h // block_size,
                           w // block_size)

@DATASET_REGISTRY.register()
class SIDDMediumRaw(data.Dataset):
    def __init__(self, opt):
        super(SIDDMediumRaw, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.data_dir = opt['dataroot']
        # get images path
        self.train_fns = glob.glob(os.path.join(self.data_dir, "*"))
        self.train_fns.sort()
        print('fetch {} samples for training'.format(len(self.train_fns)))

    def __getitem__(self, index):
        # fetch image
        fn = self.train_fns[index]
        im = loadmat(fn)["x"]
        im = im[np.newaxis, :, :]
        im = torch.from_numpy(im)
        #print(im.shape)
        return {
            'lq': im,
            'gt': im,
            'lq_path': fn
        }

    def __len__(self):
        return len(self.train_fns)