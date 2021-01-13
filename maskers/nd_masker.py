import torch
import numpy as np
from maskers.stratified_masker import StratifiedMasker
from scipy import signal
import torch.nn.functional as F


class NDMasker(StratifiedMasker):
    def __init__(self, radius, box_size):
        self.radius = radius
        self.box_size = box_size

        self.selection_dist = signal.gaussian(radius * 2 + 1, std=radius)
        self.selection_dist = self.selection_dist / self.selection_dist.sum()

    def mask_single_channel(self, X):
        device = X.device

        C, H, W = X[0].shape
        if C != 1:
            raise ValueError("number of channels must be 1")

        box_count_x = int(np.ceil(H / self.box_size))
        box_count_y = int(np.ceil(W / self.box_size))
        target_x = []
        target_y = []
        for i in range(box_count_x):
            for j in range(box_count_y):
                x, y = np.random.rand(), np.random.rand()
                x = int(i * self.box_size + x * self.box_size)
                y = int(j * self.box_size + y * self.box_size)
                if x < H and y < W:
                    target_x.append(x)
                    target_y.append(y)

        avg_kernel = torch.ones(1, 1, self.radius * 2 + 1,
                                self.radius * 2 + 1).to(device)
        avg_kernel = avg_kernel / avg_kernel.sum()
        avg_X = F.conv2d(X, avg_kernel, stride=1, padding=self.radius)

        masked_xs = []
        masked_ys = []
        for x, y in zip(target_x, target_y):
            masked_x = np.random.choice(
                np.arange(x - self.radius, x + self.radius + 1, 1),
                p=self.selection_dist
            )
            masked_y = np.random.choice(
                np.arange(y - self.radius, y + self.radius + 1, 1),
                p=self.selection_dist
            )
            if masked_x > 0 and masked_y > 0 and masked_x < H and masked_y < W:
                masked_xs.append(masked_x)
                masked_ys.append(masked_y)

        target_mask = torch.zeros((H, W)).to(device)
        target_mask[target_x, target_y] = 1.
        inv_target_mask = 1 - target_mask

        mask = torch.zeros((H, W)).to(device)
        mask[masked_xs, masked_ys] = 1.
        inv_mask = 1 - mask

        # debugging
        # out = np.ones((64, 64, 3))
        # out[:, :, 0] = target_mask.numpy()
        # out[:, :, 1] = mask.numpy()
        # import cv2
        # cv2.imwrite('debug.png', avg_X[0, 0].numpy() * 255.)

        return (avg_X * target_mask + X * inv_target_mask) * inv_mask, mask
