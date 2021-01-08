import torch
import numpy as np


class BaseMasker:
    @staticmethod
    def interpolate_mask(tensor, mask, mask_inv):
        _, num_channels, _, _ = tensor.shape
        if num_channels != 1:
            raise ValueError("number of channels must be 1")

        device = tensor.device
        mask = mask.to(device)

        kernel = np.array([[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], (0.5, 1.0, 0.5)])
        kernel = kernel[np.newaxis, np.newaxis, :, :]
        kernel = torch.Tensor(kernel).to(device)
        kernel = kernel / kernel.sum()

        filtered_tensor = torch.nn.functional.conv2d(tensor, kernel,
                                                     stride=1, padding=1)

        return filtered_tensor * mask + tensor * mask_inv

    def mask_single_channel(self, X):
        pass

    def mask(self, X):
        batchsz, C, H, W = X.shape
        mask = torch.zeros((H, W)).to(X.device)

        masked = torch.zeros((batchsz, C, H, W)).to(X.device)
        mask = torch.zeros((C, H, W)).to(X.device)

        for i in range(C):
            masked_i, mask_i = self.mask_single_channel(X[:, i: i + 1, :, :])
            masked[:, i, :, :] = masked_i[:, 0, :, :]
            mask[i, :, :] = mask_i

        return masked, mask
