from noise_models.base_noise_model import BaseNoiseModel
import numpy as np


class CombinedNoiseModel(BaseNoiseModel):
    def __init__(self, gauss_sigma, poiss_lambda, bern_p):
        self.gauss_sigma = gauss_sigma
        self.poiss_lambda = poiss_lambda
        self.bern_p = bern_p

    def add_noises(self, img):
        noised_img = 255. * np.random.poisson(self.poiss_lambda
                                              * img / 255.) / self.poiss_lambda
        noised_img = noised_img + np.random.normal(0, self.gauss_sigma,
                                                   img.shape)
        bernoulli_noise_map = np.random.binomial(1, 0.5, img.shape) * 255.
        noised_img = np.where(np.random.uniform(0, 1, img.shape) < self.bern_p,
                              bernoulli_noise_map, noised_img)

        return noised_img
