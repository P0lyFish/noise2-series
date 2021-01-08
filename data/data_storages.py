import os.path as osp
import lmdb
import numpy as np
import pickle


class BaseDataStorage:
    def get(index):
        pass

    def __len__(self):
        pass


class LmdbDataStorage(BaseDataStorage):
    def __init__(self, data_path):
        self.env = lmdb.open(
            data_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )
        with open(osp.join(osp.dirname(data_path),
                           'meta_info.pkl'), 'rb') as f:
            meta_info = pickle.load(f)

        self.img_size = list(map(int, meta_info['resolution'].split('_')))
        self.keys = meta_info['keys']
        self.dtype = meta_info['dtype']

    def __getitem__(self, key):
        key = '{:08d}'.format(key)
        with self.env.begin(write=False) as txn:
            buf = txn.get(key.encode('ascii'))

        num_pixels = self.img_size[0] * self.img_size[1] * self.img_size[2]

        if len(buf) == num_pixels * 8:
            img_flat = np.frombuffer(buf, dtype=np.float64)
        elif len(buf) == num_pixels * 1:
            img_flat = np.frombuffer(buf, dtype=np.uint8)
        else:
            print(len(buf))
            raise ValueError("Invalid data size")
        H, W, C = self.img_size
        img = img_flat.reshape(H, W, C)

        return img

    def __len__(self):
        return len(self.keys)


class NumpyDataStorage(BaseDataStorage):
    def __init__(self, data_path):
        self.data = np.load(data_path, allow_pickle=True)

    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return self.data.shape[0]
