import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.lr_scheduler as lr_scheduler
from .base_trainer import BaseTrainer
from models.loss import CharbonnierLoss
from models.unet import Unet
import numpy as np
import cv2
from utils.util import tensor2img
from mask import Masker

logger = logging.getLogger('base')


class Trainer(BaseTrainer):
    def __init__(self, opt):
        super(Trainer, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define network and load pretrained models
        self.netG = Unet(opt['network_G']['img_channels'],
                         opt['network_G']['img_channels']).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG,
                                                device_ids=[
                                                    torch.cuda.current_device()
                                                ])
        else:
            self.netG = DataParallel(self.netG)

        self.masker = Masker()

        # print network
        self.print_network()
        self.load()

        if self.is_train:
            self.netG.train()

            # loss
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss(reduction='sum').to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss(reduction='sum').to(self.device)
            elif loss_type == 'cb':
                self.cri_pix = CharbonnierLoss().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not\
                                          recognized.'.format(loss_type))

            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G']\
                else 0
            params = []
            for k, v in self.netG.named_parameters():
                if v.requires_grad:
                    params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not\
                                       optimize.'.format(k))
            optim_params = [
                {
                    'params': params,
                    'lr': train_opt['lr_G']
                },
            ]

            self.optimizer_G = torch.optim.Adam(optim_params,
                                                lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'],
                                                       train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(
                            optimizer,
                            train_opt['lr_steps'],
                            restarts=train_opt['restarts'],
                            weights=train_opt['restart_weights'],
                            gamma=train_opt['lr_gamma'],
                            clear_state=train_opt['clear_state']
                        )
                    )
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'],
                            eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'],
                            weights=train_opt['restart_weights']
                        )
                    )
            else:
                raise NotImplementedError()

            self.log_dict = OrderedDict()

    def feed_data(self, data, need_GT=True):
        self.LQ = data['LQ'].to(self.device)
        if need_GT:
            self.HQ = data['HQ'].to(self.device)
        # LQ_numpy = self.LQ[0, 0, :, :].cpu().numpy()
        # cv2.imwrite('LQ.png', LQ_numpy * 255.)


    def set_params_lr_zero(self, groups):
        # fix normal module
        for group in groups:
            self.optimizers[0].param_groups[group]['lr'] = 0

    def optimize_parameters(self, step):
        batchsz, _, _, _ = self.LQ.shape

        self.optimizer_G.zero_grad()

        inp, mask = self.masker.mask(self.LQ, step)
        out = self.netG(inp)

        l_total = self.cri_pix(out * mask, self.LQ * mask)

        l_total.backward()
        self.optimizer_G.step()

        # set log
        self.log_dict['l_total'] = l_total.item() / batchsz

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.pred = self.netG(self.LQ)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.LQ.detach()[0].float().cpu()
        out_dict['GT'] = self.HQ.detach()[0].float().cpu()
        out_dict['pred'] = self.pred.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(
                self.netG.__class__.__name__,
                self.netG.module.__class__.__name__
            )
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, \
                        with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        if self.opt['path']['pretrain_model_G']:
            load_path_G = self.opt['path']['pretrain_model_G']
            if load_path_G is not None:
                logger.info('Loading model for G [{:s}]\
                            ...'.format(load_path_G))
                self.load_network(load_path_G, self.netG,
                                  self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
