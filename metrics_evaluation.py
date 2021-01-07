import torch
import numpy as np
import os
import logging
import argparse

from models.unet import Unet
import utils.util as util


def evaluation(num_channels, model_path, save_path, LQ_path, HQ_path):
    device = torch.device('cuda')

    # Initializing logger
    logger = logging.getLogger('base')
    os.makedirs(save_path, exist_ok=True)
    util.setup_logger('base', save_path, 'test', level=logging.INFO,
                      screen=True, tofile=True)
    logger.info('LQ path: {}'.format(LQ_path))
    logger.info('HQ path: {}'.format(HQ_path))
    logger.info('model path: {}'.format(model_path))

    # Initializing mode
    logger.info('Loading model...')
    model = Unet(num_channels, num_channels).to(device)
    model.eval()
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    logger.info('Done')

    # processing data
    HQ_data = np.load(HQ_path, allow_pickle=True)
    LQ_data = np.load(LQ_path, allow_pickle=True)

    num_images = LQ_data.shape[0]
    logger.info('Number of test images: {}'.format(num_images))

    psnr_avg = 0

    resizer = util.PadAndCropResizer()

    for idx in range(num_images):
        LQ_img = LQ_data[idx] / 255.
        if LQ_img.ndim == 2:
            LQ_img = LQ_img[:, :, np.newaxis]
        LQ_img = resizer.before(LQ_img, 16, 2)
        LQ_img = np.transpose(LQ_img, (2, 0, 1))[np.newaxis, :, :, :]
        LQ_img = torch.Tensor(LQ_img).to(device)

        with torch.no_grad():
            pred_img = model(LQ_img)

        pred_img = util.tensor_to_numpy(pred_img) * 255.
        pred_img = resizer.after(pred_img, 2)

        psnr = util.calculate_psnr(HQ_data[idx], pred_img)

        print('image #{}, PSNR: {:.2f}db'.format(idx + 1, psnr))
        psnr_avg += psnr

    print('Average PSNR: {:.2f}db'.format(psnr_avg / num_images))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_channels', type=int,
                        required=True,
                        help='number of image channels')
    parser.add_argument('--model', type=str,
                        choices=['noise2true', 'noise2noise', 'noise2void',
                                 'noise2self', 'noise2same'],
                        required=True,
                        help='Denoising model')
    parser.add_argument('--pretrained_path', type=str,
                        default='',
                        help='pretrained model path')

    parser.add_argument('--save_path', type=str,
                        default='',
                        help='where to save test results')
    parser.add_argument('--LQ_path', type=str,
                        required=True,
                        help='noise data path')
    parser.add_argument('--HQ_path', type=str,
                        required=True,
                        help='clean data path')

    args = parser.parse_args()

    if args.pretrained_path == '':
        args.pretrained_path = 'pretrained_models/{}.pth'.format(args.model)
    if args.save_path == '':
        args.save_path = 'results/{}'.format(args.model)

    evaluation(args.num_channels, args.pretrained_path, args.save_path,
               args.LQ_path, args.HQ_path)
