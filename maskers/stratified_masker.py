import numpy as np
import torch
from maskers.base_masker import BaseMasker


class StratifiedMasker(BaseMasker):
    """Object for masking and demasking"""

    def __init__(self, mode='zero', box_size=None):
        self.box_size = box_size
        self.mode = mode

    @staticmethod
    def stratified_mask(X, box_size):
        H, W = X[0, 0].shape
        mask = torch.zeros((H, W))
        box_count_x = int(np.ceil(H / box_size))
        box_count_y = int(np.ceil(W / box_size))
        x_coords = []
        y_coords = []
        for i in range(box_count_x):
            for j in range(box_count_y):
                x, y = np.random.rand(), np.random.rand()
                x = int(i * box_size + x * box_size)
                y = int(j * box_size + y * box_size)
                if x < H and y < W:
                    x_coords.append(x)
                    y_coords.append(y)
        mask[x_coords, y_coords] = 1.

        return mask

    def mask_single_channel(self, X):
        mask = self.stratified_mask(X, self.box_size)

        mask = mask.to(X.device)
        mask_inv = torch.ones(mask.shape).to(X.device) - mask

        if self.mode == 'interpolate':
            masked = self.interpolate_mask(X, mask, mask_inv)
        elif self.mode == 'zero':
            masked = X * mask_inv
        else:
            raise NotImplementedError

        return masked, mask
