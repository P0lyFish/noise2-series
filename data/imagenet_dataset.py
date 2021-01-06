'''
ImageNet dataset
'''
import logging
import numpy as np
import random
import torch
import torch.utils.data as data

import data.util as util
from utils.util import PadAndCropResizer

logger = logging.getLogger('base')


class ImageNetDataset(data.Dataset):
    '''
    Reading the training ImageNet dataset
    '''

    def __init__(self, opt):
        super(ImageNetDataset, self).__init__()
        self.opt = opt
        # temporal augmentation

        self.LQ_data = np.load(opt['LQ_data'], allow_pickle=True)

        if opt['HQ_data'] is not None:
            self.HQ_data = np.load(opt['HQ_data'], allow_pickle=True)
            self.need_GT = True
        else:
            self.need_GT = False

        self.cropper = PadAndCropResizer()

        assert self.LQ_data.shape[0], 'Error: LQ data is empty.'

    def __getitem__(self, index):
        img_LQ = self.LQ_data[index]
        if self.need_GT:
            img_HQ = self.HQ_data[index]
        else:
            img_HQ = None

        img_LQ = img_LQ[:, :] / 255.
        if self.need_GT:
            img_HQ = img_HQ[:, :] / 255.

        if self.opt['phase'] == 'train':
            H, W, _ = img_LQ.shape
            HQ_size = self.opt['HQ_size']
            rnd_h = random.randint(0, max(0, H - HQ_size))
            rnd_w = random.randint(0, max(0, W - HQ_size))
            img_LQ = img_LQ[rnd_h:rnd_h + HQ_size, rnd_w:rnd_w + HQ_size, :]
            if self.need_GT:
                img_HQ = img_HQ[rnd_h:rnd_h + HQ_size,
                                rnd_w:rnd_w + HQ_size, :]

            rlt = util.augment([img_LQ, img_HQ], self.opt['use_flip'],
                               self.opt['use_rot'])
            img_LQ = rlt[0]
            img_HQ = rlt[1]

        img_LQ = np.transpose(img_LQ, (2, 0, 1))
        img_LQ = torch.from_numpy(np.ascontiguousarray(img_LQ)).float()

        if img_HQ is not None:
            img_HQ = np.transpose(img_HQ, (2, 0, 1))
            img_HQ = torch.from_numpy(np.ascontiguousarray(img_HQ)).float()

        if img_HQ is not None:
            return {'LQ': img_LQ, 'HQ': img_HQ}
        return {'LQ': img_LQ}

    def __len__(self):
        return self.LQ_data.shape[0]
