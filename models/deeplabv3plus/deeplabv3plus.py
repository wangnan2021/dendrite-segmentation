#!/usr/bin/env
# -*- coding: utf-8 -*-

'''
Copyright (c) 2018 Yude Wang
Copyright (c) 2019 Charles
'''

from __future__ import absolute_import

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

# from model.basemodules import resnet
from deeplabv3plus import efficientnet
from .aspp import ASPP, DenseASPP


__all__ = ['deeplabv3plus_xception', 'deeplabv3plus_efficientnet']

model_urls = {
    'deeplabv3plus_xception': os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'checkpoints/deeplabv3plus_xception_VOC2012_epoch46_all.pth'),
    'deeplabv3_resnet101_coco': 'https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth'
}


def build_backbone(backbone_name, pretrained=True, os=16):   # 下采16倍（efficientnet只抽了前4个block）
    # if backbone_name == 'resnet101':
    #     net = resnet.resnet101(pretrained=pretrained, os=os)
    #     return net
    if backbone_name.find('efficientnet') >= 0:
        intermediate = {
            'b7':[3, 10, 17, 37]
        }[backbone_name.split('-')[-1]]
        if pretrained:
            net = efficientnet.EfficientNetPartial.from_pretrained(
                backbone_name, intermediate=intermediate)
        else:
            net = efficientnet.EfficientNetPartial.from_name(backbone_name, intermediate=intermediate)
        return net
    else:
        raise ValueError(
            'backbone.py: The backbone named {} is not supported yet.'.format(
                backbone_name)
        )

class deeplabv3plus(nn.Module):
    def __init__(self, backbone_name='efficientnet-b7', decoder_name='aspp', aux=False, pretrained=False):
        super(deeplabv3plus, self).__init__()
        indim = {
            'resnet101': [64, 256, 1024, 2048],
            'efficientnet-b7': [32, 48, 80, 224]
        }[backbone_name]
        conv = nn.Conv2d
        if decoder_name.find('denseaspp') >= 0:
            self.aspp = DenseASPP(in_channels=indim[-1], nclass=256, conv_layer=conv)
        elif decoder_name.find('aspp') >= 0:
            self.aspp = ASPP(dim_in=indim[-1],
                             dim_out=256,
                             rate=1,
                             bn_mom=0.9,
                             conv=conv)
        self.shortcut_conv = nn.Sequential(
            conv(indim[1], 48, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(48, momentum=0.9),
            nn.ReLU(inplace=True)
        )
        if aux:
            self.aux_conv = nn.ModuleList()
            for idx in range(3):
                self.aux_conv.append(
                    nn.Sequential(conv(indim[idx], 48, 1, 1, padding=0, bias=True),
                                nn.BatchNorm2d(48, momentum=0.9),
                                nn.ReLU(inplace=True),
                                conv(48, 64, 3, 1, padding=1, bias=True),
                                nn.BatchNorm2d(64, momentum=0.9),
                                nn.ReLU(inplace=True),
                                nn.Dropout(0.5),
                                conv(64, 64, 3, 1, padding=1, bias=True),
                                nn.BatchNorm2d(64, momentum=0.9),
                                nn.ReLU(inplace=True),
                                nn.Dropout(0.1),
                                conv(64, 21, 1, 1, padding=0)
                            )
                )
        self.cat_conv = nn.Sequential(
            conv(256+48, 256, 3, 1, padding=1, bias=True),
            nn.BatchNorm2d(256, momentum=0.9),
            nn.ReLU(inplace=True),
            conv(256, 256, 3, 1, padding=1, bias=True),
            nn.BatchNorm2d(256, momentum=0.9),
            nn.ReLU(inplace=True),
        )
        self.cls_conv = conv(256, 1, 1, 1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.backbone = build_backbone(
            backbone_name, pretrained=pretrained, os=16)

    def forward(self, x):
        _ = self.backbone(x)
        layers = self.backbone.get_layers()
        feature_aspp = self.aspp(layers[-1])
        feature_shallow = self.shortcut_conv(layers[1])
        feature_aspp = F.upsample(feature_aspp, size=(feature_shallow.size(2), feature_shallow.size(3)), mode='bilinear', align_corners=True)
        aux = None
        if hasattr(self, 'aux_conv'):
            aux = {}
            for idx in range(3):
                aux['aux{}'.format(idx)] = F.interpolate(self.aux_conv[idx](layers[idx]),
                    size=x.shape[-2:], mode='bilinear', align_corners=True
                )
        feature_cat = torch.cat([feature_aspp, feature_shallow], 1)
        result = self.cat_conv(feature_cat)
        result = self.cls_conv(result)
        result = F.upsample(result, size=(x.size(2), x.size(3)), mode="bilinear", align_corners=True)
        del layers[:]
        if self.training and aux is not None:
            result = {'out': result}
            if aux is not None: result.update(aux)
        return result


# def deeplabv3plus_resnet101(decoder_name='aspp', aux=False, pretrained=False):
#     model = deeplabv3plus(backbone_name='resnet101', decoder_name=decoder_name, aux=aux, pretrained=pretrained)
#     if pretrained:
#         backbone_ckp = load_state_dict_from_url(model_urls['deeplabv3_resnet101_coco'], progress=True)
#         head_ckp = torch.load(model_urls['deeplabv3plus_xception'],
#                          map_location=lambda storage, loc: storage)
#         head_ckp = {k.replace('module.', ''): v for k, v in head_ckp.items()}
#         state_dict = {}
#         state_dict.update({
#             k:v for k, v in backbone_ckp.items() if k.startswith('backbone')    
#         })
#         state_dict.update({
#             k:v for k, v in head_ckp.items() if not k.startswith('backbone')    
#         })
#         model.load_state_dict(state_dict)
#     return model


def deeplabv3plus_efficientnet(version='b7', decoder_name='aspp', aux=False, pretrained=False):
    model = deeplabv3plus(backbone_name='-'.join(['efficientnet', version]),
        decoder_name=decoder_name, aux=aux, pretrained=pretrained)
    # if pretrained:
    #     ckp = torch.load(model_urls['deeplabv3plus_xception'],
    #                      map_location=lambda storage, loc: storage)
    #     ckp = {k.replace('module.', ''): v for k, v in ckp.items()}
    #     model_dict = model.state_dict()
    #     ckp = {k: v for k, v in ckp.items() if (k in model_dict)
    #            and (v.shape == model_dict[k].shape)}
    #     model_dict.update(ckp)
    #     model.load_state_dict(model_dict)
    return model

if __name__ == '__main__':
    model = deeplabv3plus().cuda()
    img = torch.rand(8,3,123,1049).cuda()
    with torch.no_grad():
        out = model(img)
    print(out.size())
    print(sum(p.numel() for p in model.parameters()))