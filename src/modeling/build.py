from .EvRGBmetro_vanilla import EvRGBmetro_vanilla
from .fastmetro import FastMETRO_Hand_Network
from .EvImHandNet import EvImHandNet

def build_model(cfg):
    model = None
    if cfg['model']['arch'] == 'EvImHandNet':
        model = EvImHandNet(cfg)
    elif cfg['model']['arch'] == 'EvRGBmetro_vanilla':
        model = EvRGBmetro_vanilla(cfg)
    elif cfg['model']['arch'] == 'FastMETRO':
        model = FastMETRO_Hand_Network(cfg)
    else:
        raise NotImplementedError('Model {} not implemented.'.format(cfg.MODEL.NAME))
    return model