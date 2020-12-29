'''
BSD68 dataset
support reading images from lmdb
'''
import os.path as osp
import random
import pickle
import logging
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data
import data.util as util
from utils.util import PadAndCropResizer

logger = logging.getLogger('base')


class BSD68Dataset(data.Dataset):
    '''
    Reading the training BSD68 dataset
    key example: 000_00000000
    HQ: Ground-Truth;
    LQ: Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames
    '''

    def __init__(self, opt):
        super(BSD68Dataset, self).__init__()
        self.opt = opt
        # temporal augmentation

        #### directly load image keys
        logger.info('Using cache keys: {}'.format(opt['cache_keys']))

        self.LQ_data = np.load(opt['LQ_data'], allow_pickle=True)

        if opt['HQ_data'] is not None:
            self.HQ_data = np.load(opt['HQ_data'], allow_pickle=True)
        else:
            self.HQ_data = None

        # if opt['phase'] == 'val':
        #    self.LQ_data = self.LQ_data[60:]
        #    self.HQ_data = self.HQ_data[60:]

        self.cropper = PadAndCropResizer()

        assert self.LQ_data.shape[0], 'Error: LQ data is empty.'

    def __getitem__(self, index):
        if self.opt['phase'] == 'train':
            HQ_size = self.opt['HQ_size']
            #### get the HQ image
            img_LQ = self.LQ_data[index] / 255.
            img_LQ = self.cropper.before(img_LQ, 16, None)
            img_LQ = img_LQ[:, :, np.newaxis]
            img_HQ = None
            H, W, _ = img_LQ.shape  # LQ size
            # randomly crop
            # rnd_h = random.randint(0, max(0, H - HQ_size))
            # rnd_w = random.randint(0, max(0, W - HQ_size))
            # img_LQ = img_LQ[rnd_h:rnd_h + HQ_size, rnd_w:rnd_w + HQ_size, :]
            rlt = util.augment([img_LQ], self.opt['use_flip'], self.opt['use_rot'])
            img_LQ = rlt[0]
        elif self.opt['phase'] == 'val':
            img_LQ = self.LQ_data[index] / 255.
            img_HQ = self.HQ_data[index] / 255.
            img_LQ = self.cropper.before(img_LQ, 16, None)
            img_HQ = self.cropper.before(img_HQ, 16, None)
            img_LQ = img_LQ[:, :, np.newaxis]
            img_HQ = img_HQ[:, :, np.newaxis]

        img_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))).float()
        if img_HQ is not None:
            img_HQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HQ, (2, 0, 1)))).float()

        if img_HQ is None:
          return {'LQ': img_LQ}
        return {'LQ': img_LQ, 'HQ': img_HQ}

    def __len__(self):
        return self.LQ_data.shape[0]