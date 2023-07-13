
# @Time : 2022/6/26 12:02
# -------------------------
from model.backbone.resnet import resnet50
from model.backbone.convnext import convnext_small, convnext_base, convnext_tiny
from model.backbone.resnet_ibn_a import resnet50_ibn_a, resnet101_ibn_a
from model.backbone.resnext_ibna import resnext101_ibn_a
from model.backbone.swin_transformer import swin_tiny_patch4_window7_224, swin_small_patch4_window7_224, swin_base_patch4_window12_384
from model.backbone.swinv2_transformer import swinv2_tiny_patch4_window16_256, swinv2_small_patch4_window16_256, swinv2_base_patch4_window12to24_192to384_22kto1k_ft
from model.backbone.resnest import resnest101

def build_backbone(model_name, last_stride, cfg, deploy_flag):
    if model_name == 'resnet50':
        in_plane = 2048
        base = resnet50(frozen_stage=cfg.MODEL.FROZEN)
        print('using resnet50 as a backbone')

    elif model_name == 'resnet_ibn_a_101':
        in_plane = 2048
        base = resnet101_ibn_a(last_stride=1)
        print('using resnet_ibn_a_101 as a backbone')

    elif model_name == 'resnext101_ibn_a':
        in_plane = 2048
        base = resnext101_ibn_a(last_stride=2)
        print('using resnext101_ibn_a as a backbone')

    elif model_name == 'resnest101':
        in_plane = 2048
        base = resnest101(last_stride=2)
        print('using resnest101 as a backbone')

    elif model_name == 'convnext_base':
        in_plane = 1024
        base = convnext_base()
        print('using convnext_base as a backbone')

    elif model_name == 'convnext_tiny':
        in_plane = 768
        base = convnext_tiny()
        print('using convnext_tiny as a backbone')

    elif model_name == 'convnext_small':
        in_plane = 768
        base = convnext_small()
        print('using convnext-small as a backbone')


    elif model_name == 'swin_tiny_patch4_window7_224':
        in_plane = 768
        base = swin_tiny_patch4_window7_224()
        print('using swin_tiny_patch4_window7_224 as a backbone')

    elif model_name == 'swin_small_patch4_window7_224':
        in_plane = 768
        base = swin_small_patch4_window7_224()
        print('using swin_small_patch4_window7_224 as a backbone')

    elif model_name == 'swin_base_patch4_window12_384':
        in_plane = 1024
        base = swin_base_patch4_window12_384()
        print('using swin_base_patch4_window12_384 as a backbone')

    elif model_name == 'swinv2_tiny_patch4_window16_256':
        in_plane = 768
        base = swinv2_tiny_patch4_window16_256()
        print('using swinv2_tiny_patch4_window16_256 as a backbone')

    elif model_name == 'swinv2_small_patch4_window16_256':
        in_plane = 768
        base = swinv2_small_patch4_window16_256()
        print('using swinv2_small_patch4_window16_256 as a backbone')

    elif model_name == 'swinv2_base_patch4_window12to24_192to384_22kto1k_ft':
        in_plane = 1024
        base = swinv2_base_patch4_window12to24_192to384_22kto1k_ft()
        print('using swinv2_base_patch4_window12to24_192to384_22kto1k_ft as a backbone')

    return in_plane, base
