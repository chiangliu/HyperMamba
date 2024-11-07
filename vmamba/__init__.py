import os
from functools import partial

import torch

from .vmamba import VSSM
# from .vmamba_woatt import VSSM as VSSM_woatt
# from .vmamba_wospe import VSSM as VSSM_wospe
# from .vmamba_selectnum import VSSM as VSSM_selectnum

# try:
#     from .vim import build_vim
# except Exception as e:
#     build_vim = lambda *args, **kwargs: None


# still on developing...
def build_vssm_model(config, is_pretrain=False):
    model_type = config.MODEL.TYPE
    if model_type in ["vssm"]:
        model = VSSM(
            patch_size=config.MODEL.VSSM.PATCH_SIZE, 
            in_chans=config.MODEL.VSSM.IN_CHANS, 
            num_classes=config.MODEL.NUM_CLASSES, 
            depths=config.MODEL.VSSM.DEPTHS, 
            dims=config.MODEL.VSSM.EMBED_DIM, 
            # ===================
            ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
            ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
            ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
            ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
            ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
            ssm_conv=config.MODEL.VSSM.SSM_CONV,
            ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
            ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
            ssm_init=config.MODEL.VSSM.SSM_INIT,
            forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
            # ===================
            mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
            mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
            mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
            # ===================
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            patch_norm=config.MODEL.VSSM.PATCH_NORM,
            norm_layer=config.MODEL.VSSM.NORM_LAYER,
            downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
            patchembed_version=config.MODEL.VSSM.PATCHEMBED,
            gmlp=config.MODEL.VSSM.GMLP,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
        )
        return model

    return None
#
# # still on developing...
# def build_vssm_woatt_model(config, is_pretrain=False):
#     model_type = config.MODEL.TYPE
#     if model_type in ["vssm_woatt"]:
#         model = VSSM_woatt(
#             patch_size=config.MODEL.VSSM.PATCH_SIZE,
#             in_chans=config.MODEL.VSSM.IN_CHANS,
#             num_classes=config.MODEL.NUM_CLASSES,
#             depths=config.MODEL.VSSM.DEPTHS,
#             dims=config.MODEL.VSSM.EMBED_DIM,
#             # ===================
#             ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
#             ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
#             ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
#             ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
#             ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
#             ssm_conv=config.MODEL.VSSM.SSM_CONV,
#             ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
#             ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
#             ssm_init=config.MODEL.VSSM.SSM_INIT,
#             forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
#             # ===================
#             mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
#             mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
#             mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
#             # ===================
#             drop_path_rate=config.MODEL.DROP_PATH_RATE,
#             patch_norm=config.MODEL.VSSM.PATCH_NORM,
#             norm_layer=config.MODEL.VSSM.NORM_LAYER,
#             downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
#             patchembed_version=config.MODEL.VSSM.PATCHEMBED,
#             gmlp=config.MODEL.VSSM.GMLP,
#             use_checkpoint=config.TRAIN.USE_CHECKPOINT,
#         )
#         return model
#
#     return None
#
# # still on developing...
# def build_vssm_wospe_model(config, is_pretrain=False):
#     model_type = config.MODEL.TYPE
#     if model_type in ["vssm_wospe"]:
#         model = VSSM_wospe(
#             patch_size=config.MODEL.VSSM.PATCH_SIZE,
#             in_chans=config.MODEL.VSSM.IN_CHANS,
#             num_classes=config.MODEL.NUM_CLASSES,
#             depths=config.MODEL.VSSM.DEPTHS,
#             dims=config.MODEL.VSSM.EMBED_DIM,
#             # ===================
#             ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
#             ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
#             ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
#             ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
#             ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
#             ssm_conv=config.MODEL.VSSM.SSM_CONV,
#             ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
#             ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
#             ssm_init=config.MODEL.VSSM.SSM_INIT,
#             forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
#             # ===================
#             mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
#             mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
#             mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
#             # ===================
#             drop_path_rate=config.MODEL.DROP_PATH_RATE,
#             patch_norm=config.MODEL.VSSM.PATCH_NORM,
#             norm_layer=config.MODEL.VSSM.NORM_LAYER,
#             downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
#             patchembed_version=config.MODEL.VSSM.PATCHEMBED,
#             gmlp=config.MODEL.VSSM.GMLP,
#             use_checkpoint=config.TRAIN.USE_CHECKPOINT,
#         )
#         return model
#
#     return None
#
#
# def build_vssm_selectnum_model(config, is_pretrain=False):
#     model_type = config.MODEL.TYPE
#     if model_type in ["vssm_selectnum"]:
#         model = VSSM_selectnum(
#             patch_size=config.MODEL.VSSM.PATCH_SIZE,
#             in_chans=config.MODEL.VSSM.IN_CHANS,
#             num_classes=config.MODEL.NUM_CLASSES,
#             depths=config.MODEL.VSSM.DEPTHS,
#             dims=config.MODEL.VSSM.EMBED_DIM,
#             # ===================
#             ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
#             ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
#             ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
#             ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
#             ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
#             ssm_conv=config.MODEL.VSSM.SSM_CONV,
#             ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
#             ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
#             ssm_init=config.MODEL.VSSM.SSM_INIT,
#             forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
#             # ===================
#             mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
#             mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
#             mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
#             # ===================
#             drop_path_rate=config.MODEL.DROP_PATH_RATE,
#             patch_norm=config.MODEL.VSSM.PATCH_NORM,
#             norm_layer=config.MODEL.VSSM.NORM_LAYER,
#             downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
#             patchembed_version=config.MODEL.VSSM.PATCHEMBED,
#             gmlp=config.MODEL.VSSM.GMLP,
#             use_checkpoint=config.TRAIN.USE_CHECKPOINT,
#         )
#         return model
#
#     return None


def build_model(config, is_pretrain=False):
    model = None
    if model is None:
        model = build_vssm_model(config, is_pretrain)
    # if model is None:
    #     model = build_vssm_woatt_model(config, is_pretrain)
    # if model is None:
    #     model = build_vssm_wospe_model(config, is_pretrain)
    # if model is None:
    #     model = build_vssm_selectnum_model(config, is_pretrain)
    return model




