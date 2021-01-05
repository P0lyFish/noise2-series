class Masker:
    def mask(self, X):
        pass

    @staticmethod
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
