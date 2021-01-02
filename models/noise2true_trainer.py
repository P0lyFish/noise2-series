import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.lr_scheduler as lr_scheduler
from .base_trainer import BaseTrainer
from models.loss import CharbonnierLoss
from models.unet import Unet

logger = logging.getLogger('base')


class Noise2TrueTrainer(BaseTrainer):
    def __init__(self, opt):
        super(Noise2TrueTrainer, self).__init__(opt)

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

    def optimize_parameters(self, step):
        batchsz, _, _, _ = self.LQ.shape

        self.optimizer_G.zero_grad()

        out = self.netG(self.LQ)
        l_total = self.cri_pix(out, self.LQ)

        l_total.backward()
        self.optimizer_G.step()

        # set log
        self.log_dict['l_total'] = l_total.item() / batchsz
