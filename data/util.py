import os.path as osp
import random

from data.data_storages import LmdbDataStorage, NumpyDataStorage

####################
# Files & IO
####################

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png',
                  '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def load_data_storage(data_path):
    if data_path is None:
        return None

    if osp.splitext(data_path)[1] == '.npy':
        return NumpyDataStorage(data_path)
    elif osp.splitext(data_path)[1] == '.lmdb':
        return LmdbDataStorage(data_path)
    else:
        raise NotImplementedError('Unrecognized datatype')


####################
# image processing
# process on numpy image
####################


def augment(img_list, hflip=True, rot=True):
    """horizontal flip OR rotate (0, 90, 180, 270 degrees)"""
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if img is None:
            return img

        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]


def augment_flow(img_list, flow_list, hflip=True, rot=True):
    """horizontal flip OR rotate (0, 90, 180, 270 degrees) with flows"""
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflip:
            flow = flow[:, ::-1, :]
            flow[:, :, 0] *= -1
        if vflip:
            flow = flow[::-1, :, :]
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    rlt_img_list = [_augment(img) for img in img_list]
    rlt_flow_list = [_augment_flow(flow) for flow in flow_list]

    return rlt_img_list, rlt_flow_list
