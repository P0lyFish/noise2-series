import numpy as np
import torch


class PhaseMasker():
    """Object for masking and demasking"""

    def __init__(self, width=3, mode='interpolate'):
        self.grid_size = width
        self.n_masks = width ** 2

        self.mode = mode

    @staticmethod
    def pixel_grid_mask(shape, patch_size, phase_x, phase_y):
        A = torch.zeros(shape[-2:])
        for i in range(shape[-2]):
            for j in range(shape[-1]):
                if (i % patch_size == phase_x and j % patch_size == phase_y):
                    A[i, j] = 1
        return torch.Tensor(A)

    def mask(self, X):
        i = np.random.randint(1000)
        phasex = i % self.grid_size
        phasey = (i // self.grid_size) % self.grid_size
        mask = self.pixel_grid_mask(X[0, 0].shape, self.grid_size,
                               phasex, phasey)

        mask = mask.to(X.device)
        mask_inv = torch.ones(mask.shape).to(X.device) - mask

        if self.mode == 'interpolate':
            masked = self.interpolate_mask(X, mask, mask_inv)
        elif self.mode == 'zero':
            masked = X * mask_inv
        else:
            raise NotImplementedError

        return masked, mask

    def __len__(self):
        return self.n_masks
