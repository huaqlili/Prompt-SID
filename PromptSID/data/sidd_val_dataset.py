from torch.utils import data as data
from torchvision.transforms.functional import normalize

from PromptSID.data.data_util import (paired_paths_from_folder,
                                    paired_DP_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file)
from PromptSID.data.transforms import augment, paired_random_crop, paired_random_crop_DP, random_augmentation
from PromptSID.utils import FileClient, imfrombytes, img2tensor, padding, padding_DP, imfrombytesDP
from basicsr.utils.registry import DATASET_REGISTRY
from scipy.io import loadmat, savemat
import os
import random
import numpy as np
import torch
import cv2

@DATASET_REGISTRY.register()
class SIDDvalDataset(data.Dataset):
    def __init__(self, opt):
        super(SIDDvalDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        
        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        val_data_dict = loadmat(os.path.join(self.lq_folder, "ValidationNoisyBlocksRaw.mat"))
        val_data_noisy = val_data_dict['ValidationNoisyBlocksRaw']
        self.noisy = np.reshape(val_data_noisy, (1280,256,256))

        val_data_dict = loadmat(os.path.join(self.gt_folder, 'ValidationGtBlocksRaw.mat'))
        val_data_gt = val_data_dict['ValidationGtBlocksRaw']
        self.im = np.reshape(val_data_gt, (1280,256,256))

        print('fetch {} pairs for training'.format(len(self.noisy)))

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)
        index = index % len(self.noisy)

        img_lq = self.noisy[index]
        img_lq = img_lq[np.newaxis, :, :]
        img_lq = torch.from_numpy(img_lq)

        img_gt = self.im[index]
        img_gt = img_gt[np.newaxis, :, :]
        img_gt = torch.from_numpy(img_gt)
        
        return {
            'lq': img_lq,
            'gt': img_gt
        }

    def __len__(self):
        return len(self.noisy)