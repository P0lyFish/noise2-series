import logging
logger = logging.getLogger('base')


def get_trainer(opt):
    model = opt['trainer']
    # image restoration
    if model == 'noise2self':
        from .noise2self_trainer import Noise2SelfTrainer as M
    elif model == 'noise2true':
        from .noise2true_trainer import Noise2TrueTrainer as M
    elif model == 'noise2same':
        from .noise2same_trainer import Noise2SameTrainer as M
    else:
        raise NotImplementedError('Trainer [{:s}] not recognized.'.format(model))
    m = M(opt)
    logger.info('Trainer [{:s}] is created.'.format(m.__class__.__name__))
    return m
