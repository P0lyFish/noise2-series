import os
import logging
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel


logger = logging.getLogger('base')


class BaseTrainer():
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.is_train = opt['is_train']
        self.schedulers = []
        self.optimizers = []

    def feed_data(self, data, need_GT=True):
        self.LQ = data['LQ'].to(self.device)
        if need_GT:
            self.HQ = data['HQ'].to(self.device)

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.LQ.detach()[0].float().cpu()
        out_dict['GT'] = self.HQ.detach()[0].float().cpu()
        out_dict['pred'] = self.pred.detach()[0].float().cpu()
        return out_dict

    def get_current_log(self):
        return self.log_dict

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
        pass

    def _set_lr(self, lr_groups_l):
        """Set learning rate for warmup
        lr_groups_l: list for lr_groups. each for a optimizer"""
        for optimizer, lr_groups in zip(self.optimizers, lr_groups_l):
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group['lr'] = lr

    def _get_init_lr(self):
        """Get the initial lr, which is set by the scheduler"""
        init_lr_groups_l = []
        for optimizer in self.optimizers:
            init_lr_groups_l.append([v['initial_lr'] for v in optimizer.param_groups])
        return init_lr_groups_l

    def update_learning_rate(self, cur_iter, warmup_iter=-1):
        for scheduler in self.schedulers:
            scheduler.step()
        # set up warm-up learning rate
        if cur_iter < warmup_iter:
            # get initial lr for each group
            init_lr_g_l = self._get_init_lr()
            # modify warming-up learning rates
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append([v / warmup_iter * cur_iter for v in init_lr_g])
            # set learning rate
            self._set_lr(warm_up_lr_l)

    def get_current_learning_rate(self):
        return [param_group['lr'] for param_group in self.optimizers[0].param_groups]

    def get_network_description(self, network):
        """Get the string and total parameters of the network"""
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        return str(network), sum(map(lambda x: x.numel(), network.parameters()))

    def save_network(self, network, network_label, iter_label):
        save_filename = '{}_{}.pth'.format(iter_label, network_label)
        save_path = os.path.join(self.opt['path']['models'], save_filename)
        # print('xxx {}'.format(len(list(network.parameters()))))
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def load_network(self, load_path, network, strict=True, prefix=''):
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        load_net = torch.load(load_path)
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        load_net.update(load_net_clean)

        model_dict = network.state_dict()
        for k, v in load_net.items():
            k = prefix + k
            if (k in model_dict) and (v.shape == model_dict[k].shape):
                model_dict[k] = v
            else:
                print('Load failed:', k)

        network.load_state_dict(model_dict, strict=True)

    def save_training_state(self, epoch, iter_step):
        """Save training state during training, which will be used for resuming"""
        state = {'epoch': epoch, 'iter': iter_step, 'schedulers': [], 'optimizers': []}
        for s in self.schedulers:
            state['schedulers'].append(s.state_dict())
        for o in self.optimizers:
            state['optimizers'].append(o.state_dict())
        save_filename = '{}.state'.format(iter_step)
        save_path = os.path.join(self.opt['path']['training_state'], save_filename)
        torch.save(state, save_path)

    def resume_training(self, resume_state):
        """Resume the optimizers and schedulers for training"""
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        assert len(resume_optimizers) == len(self.optimizers), 'Wrong lengths of optimizers'
        assert len(resume_schedulers) == len(self.schedulers), 'Wrong lengths of schedulers'
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.pred = self.netG(self.LQ)
        self.netG.train()
