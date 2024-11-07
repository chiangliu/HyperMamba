from yacs.config import CfgNode as CN
import yaml
import os

_C = CN()
_C.MODEL = CN()
_C.MODEL.TYPE = 'vssm'
_C.MODEL.NAME = 'vssm_tiny_224'
_C.MODEL.PRETRAINED = ''
_C.MODEL.RESUME = ''
_C.MODEL.NUM_CLASSES = 16
_C.MODEL.DROP_RATE = 0.0
_C.MODEL.DROP_PATH_RATE = 0.1
_C.MODEL.LABEL_SMOOTHING = 0.1
_C.MODEL.MMCKPT = False

_C.MODEL.VSSM = CN()
_C.MODEL.VSSM.PATCH_SIZE = 4
_C.MODEL.VSSM.IN_CHANS = 3
_C.MODEL.VSSM.DEPTHS = [1, 1, 1, 1]
_C.MODEL.VSSM.EMBED_DIM = 96
_C.MODEL.VSSM.SSM_D_STATE = 16
_C.MODEL.VSSM.SSM_RATIO = 2.0
_C.MODEL.VSSM.SSM_RANK_RATIO = 2.0
_C.MODEL.VSSM.SSM_DT_RANK = "auto"
_C.MODEL.VSSM.SSM_ACT_LAYER = "silu"
_C.MODEL.VSSM.SSM_CONV = 3
_C.MODEL.VSSM.SSM_CONV_BIAS = True
_C.MODEL.VSSM.SSM_DROP_RATE = 0.0
_C.MODEL.VSSM.SSM_INIT = "v0"
_C.MODEL.VSSM.SSM_FORWARDTYPE = "v2"
_C.MODEL.VSSM.MLP_RATIO = 4.0
_C.MODEL.VSSM.MLP_ACT_LAYER = "gelu"
_C.MODEL.VSSM.MLP_DROP_RATE = 0.0
_C.MODEL.VSSM.PATCH_NORM = True
_C.MODEL.VSSM.NORM_LAYER = "ln"
_C.MODEL.VSSM.DOWNSAMPLE = "v2"
_C.MODEL.VSSM.PATCHEMBED = "v2"
_C.MODEL.VSSM.GMLP = False

_C.TRAIN = CN()
_C.TRAIN.USE_CHECKPOINT = False


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        print(cfg)
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args)

    config.defrost()

    config.freeze()




# def get_config(args):
#     """Get a yacs CfgNode object with default values."""
#     # Return a clone so that the defaults will not be altered
#     # This is for the "local variable" use pattern
#     config = _C.clone()
#     config.merge_from_file(args)
#     print(config)
#     return config

def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)
    return config