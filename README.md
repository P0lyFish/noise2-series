# Noise2 series - Reproducing results of noise2-series denoisers
This repository is an unofficial pytorch implementation of noise2-series denoisers, including noise2noise, noise2void, noise2self and noise2same.


## Dependencies and Installation
* Python >= 3.7
* Pytorch >= 1.4.0
* CUDA >= 10.0

## Usage
### Training
You can train your own model using the following command:
```
python train.py -opt path_to_yaml_file
```
where `path_to_yaml_file` is the path to yaml file that contain training configurations. You can find some default configurations in `options` folder. Checkpoints and logs will be saved in `../experiments/modelName`

### Testing
To test your trained model, use:
```
python metrics_evaluation.py --model model_type --LQ_path path_to_noise --HQ_path path_to_clean
```
where `model_type` can be one of `[noise2true, noise2noise, noise2void, noise2self, noise2same]`, `LQ_path` and `LQ_path` are the path to the clean and noise data respectively.

## Important notes:
The training code is borrowed from EDVR project: https://github.com/xinntao/EDVR

## References
[1] Lehtinen, Jaakko, et al. "Noise2noise: Learning image restoration without clean data." arXiv preprint arXiv:1803.04189 (2018).
[2] Krull, Alexander, Tim-Oliver Buchholz, and Florian Jug. "Noise2void-learning denoising from single noisy images." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.
[3] Batson, Joshua, and Loic Royer. "Noise2self: Blind denoising by self-supervision." arXiv preprint arXiv:1901.11365 (2019).
[4] Xie, Yaochen, Zhengyang Wang, and Shuiwang Ji. "Noise2Same: Optimizing A Self-Supervised Bound for Image Denoising." arXiv preprint arXiv:2010.11971 (2020).
