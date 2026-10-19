# Copyright (c) OpenMMLab. All rights reserved.
#
# This release keeps the upstream MMPose backbones that are present in the
# source tree, plus the CLIP backbone/text encoders required by HPP.
from .csp_darknet import CSPDarknet
from .hrformer import HRFormer
from .hrnet import HRNet
from .resnet import ResNet
from .clip import (CLIPResNetWithAttention, CLIPTextContextEncoder,
                   CLIPTextContextEncoder1, CLIPTextencoder,
                   CLIPVisionTransformer)

__all__ = [
    'HRNet', 'ResNet', 'HRFormer', 'CSPDarknet',
    'CLIPResNetWithAttention', 'CLIPVisionTransformer', 'CLIPTextencoder',
    'CLIPTextContextEncoder', 'CLIPTextContextEncoder1'
]
