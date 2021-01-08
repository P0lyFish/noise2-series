from noise_models.base_noise_model import BaseNoiseModel
import numpy as np


class GaussianNoiseModel(BaseNoiseModel):
    def __init__(self, sigma):
        self.sigma = sigma

    def add_noises(self, img):
        return img + np.random.normal(0, self.sigma, img.shape)
