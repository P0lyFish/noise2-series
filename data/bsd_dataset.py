'''
BSD68 dataset
support reading images from lmdb
'''
import logging
import numpy as np
import torch
import torch.utils.data as data

import data.util as util
from utils.util import PadAndCropResizer

logger = logging.getLogger('base')


class BSD68Dataset(data.Dataset):
    '''
    Reading the training BSD68 dataset
    '''

    def __init__(self, opt):
        super(BSD68Dataset, self).__init__()
        self.opt = opt
        # temporal augmentation

        self.LQ_data = np.load(opt['LQ_data'], allow_pickle=True)

        if opt['HQ_data'] is not None:
            self.HQ_data = np.load(opt['HQ_data'], allow_pickle=True)
            self.need_GT = True
        else:
            self.need_GT = False

        self.mean = opt['mean']
        self.std = opt['std']

        # if opt['phase'] == 'val':
        #    self.LQ_data = self.LQ_data[60:]
        #    self.HQ_data = self.HQ_data[60:]

        self.cropper = PadAndCropResizer()

        assert self.LQ_data.shape[0], 'Error: LQ data is empty.'

    def __getitem__(self, index):
        img_LQ = self.LQ_data[index]
        img_LQ = self.cropper.before(img_LQ, 16, None)
        img_LQ = img_LQ[:, :, np.newaxis]

        if self.need_GT:
            img_HQ = self.HQ_data[index]
            img_HQ = self.cropper.before(img_HQ, 16, None)
            img_HQ = img_HQ[:, :, np.newaxis]
        else:
            img_HQ = None

        if self.opt['phase'] == 'train':
            rlt = util.augment([img_LQ, img_HQ], self.opt['use_flip'],
                               self.opt['use_rot'])
            img_LQ = rlt[0]
            img_HQ = rlt[1]

        img_LQ = np.transpose(img_LQ, (2, 0, 1))
        img_LQ = torch.from_numpy(np.ascontiguousarray(img_LQ)).float()
        img_LQ = (img_LQ - self.mean) / self.std

        if img_HQ is not None:
            img_HQ = np.transpose(img_HQ, (2, 0, 1))
            img_HQ = torch.from_numpy(np.ascontiguousarray(img_HQ)).float()
            img_HQ = (img_HQ - self.mean) / self.std

        if img_HQ is not None:
            return {'LQ': img_LQ, 'HQ': img_HQ}
        return {'LQ': img_LQ}

    def __len__(self):
        return self.LQ_data.shape[0]
