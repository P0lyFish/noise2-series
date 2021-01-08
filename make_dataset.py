import numpy as np
import cv2
import lmdb
import glob
import os
import os.path as osp
import pickle

from noise_models.combined_noise_model import CombinedNoiseModel


def cropping(img, patch_size):
    H, W, C = img.shape

    try:
        rnd_h = np.random.randint(H - patch_size[0] + 1)
        rnd_w = np.random.randint(W - patch_size[1] + 1)
    except ValueError:
        print(H, W, C, patch_size)
        raise ValueError

    return img[rnd_h: rnd_h + patch_size[0], rnd_w: rnd_w + patch_size[1], :]


def get_random_img(img_paths, patch_size):
    while True:
        img_path = np.random.choice(img_paths)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[2] != 3 or img.shape[0] < patch_size[0]\
                or img.shape[1] < patch_size[1]:
            continue

        img = cropping(img, patch_size)
        return img


def make_dataset(img_paths, patch_size, dataset_size, noise_model,
                 save_paths, targets):
    img = get_random_img(img_paths, patch_size)
    meta_save_path = osp.dirname(save_paths[0])
    batchsz = 5000
    for save_path in save_paths:
        os.makedirs(osp.dirname(save_path), exist_ok=True)

    envs = []
    for target, save_path in zip(targets, save_paths):
        envs.append(lmdb.open(save_path, map_size=img.nbytes * dataset_size *
                              10))
    txns = [env.begin(write=True) for env in envs]

    keys = []
    for idx in range(dataset_size):
        img = get_random_img(img_paths, patch_size)
        key = '{:08d}'.format(idx).encode('ascii')
        keys.append(key)
        for target, txn in zip(targets, txns):
            if target == 'clean':
                txn.put(key, img.copy(order='C'))
            elif target == 'noise':
                txn.put(key, noise_model.add_noises(img).copy(order='C'))

        if idx % batchsz == 0:
            print('Processed {}/{}'.format(idx, dataset_size))
            for txn in txns:
                txn.commit()
            txns = []
            for env in envs:
                txns.append(env.begin(write=True))

    for txn in txns:
        txn.commit()
    for env in envs:
        env.close()

    meta_info = {}
    meta_info['resolution'] = '{}_{}_{}'.format(img.shape[0], img.shape[1],
                                                img.shape[2])
    meta_info['keys'] = keys
    pickle.dump(meta_info, open(osp.join(meta_save_path,
                                         'meta_info.pkl'), "wb"))
    print('Finish creating lmdb meta info.')


if __name__ == '__main__':
    img_paths = sorted(glob.glob('/root/myImageNet/org/*.JPEG'))
    patch_size = (128, 128)
    dataset_size = 1000
    noise_model = CombinedNoiseModel(gauss_sigma=60, poiss_lambda=30.,
                                     bern_p=0.2)
    save_paths = ['datasets/ImageNet/val/clean.lmdb',
                  'datasets/ImageNet/val/noise.lmdb']
    targets = ['clean', 'noise']

    make_dataset(img_paths, patch_size, dataset_size, noise_model,
                 save_paths, targets)
