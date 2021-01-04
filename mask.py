import numpy as np
import torch


class Masker():
    """Object for masking and demasking"""

    def __init__(self, sample_method, width=3, mode='zero',
                 box_size=None,
                 infer_single_pass=False,
                 include_mask_as_input=False):
        self.grid_size = width
        self.n_masks = width ** 2
        self.box_size = box_size
        self.sample_method = sample_method

        self.mode = mode
        self.infer_single_pass = infer_single_pass
        self.include_mask_as_input = include_mask_as_input

    def mask(self, X, i=None):
        if self.sample_method == 'phase':
            phasex = i % self.grid_size
            phasey = (i // self.grid_size) % self.grid_size
            mask = pixel_grid_mask(X[0, 0].shape, self.grid_size,
                                   phasex, phasey)
        elif self.sample_method == 'stratified':
            mask = stratified_mask(X, self.box_size)
        else:
            raise NotImplementedError

        mask = mask.to(X.device)
        mask_inv = torch.ones(mask.shape).to(X.device) - mask

        if self.mode == 'interpolate':
            masked = interpolate_mask(X, mask, mask_inv)
        elif self.mode == 'zero':
            masked = X * mask_inv
        else:
            raise NotImplementedError

        if self.include_mask_as_input:
            net_input = torch.cat((masked,
                                   mask.repeat(X.shape[0], X.shape[1], 1, 1)), dim=1)
        else:
            net_input = masked

        return net_input, mask

    def __len__(self):
        return self.n_masks

    def infer_full_image(self, X, model):
        if self.infer_single_pass:
            if self.include_mask_as_input:
                net_input = torch.cat((X, torch.zeros(X[:, 0:1].shape).to(X.device)), dim=1)
            else:
                net_input = X
            net_output = model(net_input)
            return net_output

        else:
            net_input, mask = self.mask(X, 0)
            net_output = model(net_input)

            acc_tensor = torch.zeros(net_output.shape).cpu()

            for i in range(self.n_masks):
                net_input, mask = self.mask(X, i)
                net_output = model(net_input)
                acc_tensor = acc_tensor + (net_output * mask).cpu()

            return acc_tensor


def pixel_grid_mask(shape, patch_size, phase_x, phase_y):
    A = torch.zeros(shape[-2:])
    for i in range(shape[-2]):
        for j in range(shape[-1]):
            if (i % patch_size == phase_x and j % patch_size == phase_y):
                A[i, j] = 1
    return torch.Tensor(A)


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
            x = int(i * box_size + x)
            y = int(j * box_size + y)
            if x < H and y < W:
                x_coords.append(x)
                y_coords.append(y)
    mask[x_coords, y_coords] = 1.

    return mask


def interpolate_mask(tensor, mask, mask_inv):
    _, num_channels, _, _ = tensor.shape
    device = tensor.device

    mask = mask.to(device)

    kernel = np.array([[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], (0.5, 1.0, 0.5)])
    kernel = kernel[np.newaxis, np.newaxis, :, :]
    kernel = np.tile(kernel, (1, num_channels, 1, 1))
    kernel = torch.Tensor(kernel).to(device)
    kernel = kernel / kernel.sum()

    filtered_tensor = torch.nn.functional.conv2d(tensor, kernel, stride=1, padding=1)

    return filtered_tensor * mask + tensor * mask_inv
