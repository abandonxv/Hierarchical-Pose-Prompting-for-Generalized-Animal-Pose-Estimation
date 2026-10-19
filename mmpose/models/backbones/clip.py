from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from timm.models.layers import drop, drop_path, trunc_normal_
from mmengine.logging import MMLogger
#from mmpose.models.utils import load_checkpoint, load_checkpoint_clip
from mmpose.registry import MODELS
from ..builder import BACKBONES, HEADS
import math
import pdb
from functools import partial
from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from timm.models.layers import drop, drop_path, trunc_normal_
from mmengine.logging import MMLogger
from mmengine.runner import load_checkpoint as mmengine_load_checkpoint
from mmpose.registry import MODELS

import math
import pdb
from functools import partial


def load_checkpoint(model,
                    filename,
                    map_location='cpu',
                    strict=False,
                    logger=None):
    checkpoint = mmengine_load_checkpoint(model, filename, map_location, strict=strict, logger=logger)
    return checkpoint


def load_checkpoint_clip(model,
                         filename,
                         map_location='cpu',
                         strict=False,
                         logger=None):
    checkpoint = torch.load(filename, map_location=map_location)
    if 'state_dict' in checkpoint:
        state_dict_tmp = checkpoint['state_dict']
    else:
        state_dict_tmp = checkpoint

    state_dict = OrderedDict()
    for k, v in state_dict_tmp.items():
        if 'attnpool.positional_embedding' in k:
            if model.attnpool.positional_embedding.shape != v.shape:
                print(f'Resize the pos_embed shape from {v.shape} to {model.attnpool.positional_embedding.shape}')
                cls_pos = v[0:1, :]
                if model.same_dim:
                    H = W = model.input_resolution // 32
                else:
                    H = model.input_resolution // 32
                    W = model.short_dim // 32
                old_h = 8
                old_w = 6
                spatial_pos = F.interpolate(
                    v[1:, ].reshape(1, old_h, old_w, cls_pos.shape[1]).permute(0, 3, 1, 2),
                    size=(H, W), mode='bilinear')
                spatial_pos = spatial_pos.reshape(cls_pos.shape[1], H * W).permute(1, 0)
                positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)
                v = positional_embedding
                assert model.attnpool.positional_embedding.shape == v.shape
        elif 'backbone.positional_embedding' in k:
            if model.positional_embedding.shape != v.shape:
                print(f'Resize the pos_embed shape from {v.shape} to {model.positional_embedding.shape}')
                cls_pos = v[0:1, :]
                if model.same_dim:
                    H = W = model.input_resolution // model.patch_size
                else:
                    H = model.input_resolution // model.patch_size
                    W = model.short_dim // model.patch_size
                old_h = 16
                old_w = 12
                spatial_pos = F.interpolate(
                    v[1:, ].reshape(1, old_h, old_w, 768).permute(0, 3, 1, 2),
                    size=(H, W), mode='bilinear')
                spatial_pos = spatial_pos.reshape(768, H * W).permute(1, 0)
                positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)
                v = positional_embedding
                assert model.positional_embedding.shape == v.shape

        if k.startswith('module.backbone.'):
            state_dict[k[16:]] = v
        elif k.startswith('module.'):
            state_dict[k[7:]] = v
        elif k.startswith('backbone.'):
            state_dict[k[9:]] = v
        else:
            state_dict[k] = v

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
    if logger is not None:
        if missing_keys:
            logger.warning(f'Missing keys: {missing_keys}')
        if unexpected_keys:
            logger.warning(f'Unexpected keys: {unexpected_keys}')

    return checkpoint


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None, same_dim: bool = True,
                 short_dim: int = 224):
        super().__init__()
        if same_dim:
            self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        else:
            self.positional_embedding = nn.Parameter(
                torch.randn(spacial_dim * short_dim + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.spacial_dim = spacial_dim
        self.same_dim = same_dim
        self.short_dim = short_dim

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC

        cls_pos = self.positional_embedding[0:1, :]
        if self.same_dim:
            spatial_pos = F.interpolate(
                self.positional_embedding[1:, ].reshape(1, self.spacial_dim, self.spacial_dim, self.embed_dim).permute(
                    0, 3, 1, 2), size=(H, W), mode='bilinear')
        else:
            spatial_pos = F.interpolate(
                self.positional_embedding[1:, ].reshape(1, self.spacial_dim, self.short_dim, self.embed_dim).permute(
                    0, 3, 1, 2), size=(H, W), mode='bilinear')
        spatial_pos = spatial_pos.reshape(self.embed_dim, H * W).permute(1, 0)
        positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)

        x = x + positional_embedding[:, None, :]
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        x = x.permute(1, 2, 0)
        global_feat = x[:, :, 0]
        feature_map = x[:, :, 1:].reshape(B, -1, H, W)
        return global_feat, feature_map


@HEADS.register_module()
class AttentionPoolHead(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.spacial_dim = spacial_dim

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC

        cls_pos = self.positional_embedding[0:1, :]
        spatial_pos = F.interpolate(
            self.positional_embedding[1:, ].reshape(1, self.spacial_dim, self.spacial_dim, self.embed_dim).permute(0, 3,
                                                                                                                   1,
                                                                                                                   2),
            size=(H, W), mode='bilinear')
        spatial_pos = spatial_pos.reshape(self.embed_dim, H * W).permute(1, 0)
        positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)

        x = x + positional_embedding[:, None, :]
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        x = x.permute(1, 2, 0)
        global_feat = x[:, :, 0]
        feature_map = x[:, :, 1:].reshape(B, -1, H, W)
        return [global_feat, feature_map]

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()

            state_dict = {}

            for k in checkpoint.keys():
                if k.startswith('visual.'):
                    new_k = k.replace('visual.', '')
                    if 'attnpool' in new_k:
                        new_k = new_k.replace('attnpool.', '')  # TODO need further check
                        state_dict[new_k] = checkpoint[k]

                        if 'positional_embedding' in new_k:
                            if self.positional_embedding.shape != state_dict[new_k].shape:
                                print(
                                    f'Resize the pos_embed shape from {state_dict[new_k].shape} to {self.positional_embedding.shape}')
                                cls_pos = state_dict[new_k][0:1, :]
                                H = W = self.spacial_dim
                                old_h = int(math.sqrt(state_dict[new_k][1:, ].shape[0]))
                                spatial_pos = F.interpolate(
                                    state_dict[new_k][1:, ].reshape(1, old_h, old_h, cls_pos.shape[1]).permute(0, 3, 1,
                                                                                                               2),
                                    size=(H, W), mode='bilinear')
                                spatial_pos = spatial_pos.reshape(cls_pos.shape[1], H * W).permute(1, 0)
                                positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)
                                state_dict[new_k] = positional_embedding
                                assert self.positional_embedding.shape == state_dict[new_k].shape

            u, w = self.load_state_dict(state_dict, False)
            print(u, w, 'are misaligned params in CLIPResNet AttnPool')


@MODELS.register_module()
class CLIPResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim=512, input_resolution=224, width=64, pretrained=None, **kwargs):
        super().__init__()
        self.pretrained = pretrained
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()

            state_dict = {}

            for k in checkpoint.keys():
                if k.startswith('visual.'):
                    new_k = k.replace('visual.', '')
                    state_dict[new_k] = checkpoint[k]

            u, w = self.load_state_dict(state_dict, False)
            print(u, w, 'are misaligned params in CLIPResNet')

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)

        outs = []
        x = self.layer1(x)
        outs.append(x)
        x = self.layer2(x)
        outs.append(x)
        x = self.layer3(x)
        outs.append(x)
        x = self.layer4(x)
        outs.append(x)

        return tuple(outs)


@BACKBONES.register_module()
class CLIPResNetWithAttention(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim=1024, input_resolution=224, width=64, pretrained=None, same_dim=True,
                 short_dim=224, **kwargs):
        super().__init__()
        self.pretrained = pretrained
        self.output_dim = output_dim
        self.input_resolution = input_resolution
        self.same_dim = same_dim
        self.short_dim = short_dim

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, 32, output_dim, same_dim=same_dim,
                                        short_dim=short_dim // 32)

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()

            state_dict = {}

            for k in checkpoint.keys():
                if k.startswith('visual.'):
                    new_k = k.replace('visual.', '')
                    state_dict[new_k] = checkpoint[k]

                    if 'positional_embedding' in new_k:
                        if self.attnpool.positional_embedding.shape != state_dict[new_k].shape:
                            print(
                                f'Resize the pos_embed shape from {state_dict[new_k].shape} to {self.attnpool.positional_embedding.shape}')
                            cls_pos = state_dict[new_k][0:1, :]
                            if self.same_dim:
                                H = W = self.input_resolution // 32
                            else:
                                H = self.input_resolution // 32  # TODO H is longer
                                W = self.short_dim // 32
                            old_h = int(math.sqrt(state_dict[new_k][1:, ].shape[0]))
                            spatial_pos = F.interpolate(
                                state_dict[new_k][1:, ].reshape(1, old_h, old_h, cls_pos.shape[1]).permute(0, 3, 1, 2),
                                size=(H, W), mode='bilinear')
                            spatial_pos = spatial_pos.reshape(cls_pos.shape[1], H * W).permute(1, 0)
                            positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)
                            state_dict[new_k] = positional_embedding
                            assert self.attnpool.positional_embedding.shape == state_dict[new_k].shape

            u, w = self.load_state_dict(state_dict, False)
            print(u, w, 'are misaligned params in CLIPResNet')

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        outs = []

        x = stem(x)
        x = self.layer1(x)
        outs.append(x)
        x = self.layer2(x)
        outs.append(x)
        x = self.layer3(x)
        outs.append(x)
        x = self.layer4(x)
        outs.append(x)

        x_global, x_local = self.attnpool(x)
        outs.append([x_global, x_local])

        return tuple(outs)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, drop_path=0.):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.drop_path(self.attention(self.ln_1(x)))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, drop_path_rate=0.):
        super().__init__()
        self.width = width
        self.layers = layers
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, layers)]  # stochastic depth decay rule
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask, dpr[i]) for i in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class ResidualAttentionBlock_WithoutFFN(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, drop_path=0.):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.drop_path(self.attention(self.ln_1(x)))
        return x


class Transformer_WithoutFFN(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, drop_path_rate=0.):
        super().__init__()
        self.width = width
        self.layers = layers
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, layers)]  # stochastic depth decay rule
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock_WithoutFFN(width, heads, attn_mask, dpr[i]) for i in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N, C = q.shape
        assert k.shape == v.shape
        B, M, C = k.shape
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads)
        k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads)
        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads)

        attn = torch.einsum('bnkc,bmkc->bknm', q, k) * self.scale

        attn = attn.softmax(dim=-1)

        x = torch.einsum('bknm,bmkc->bnkc', attn, v).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(
            self,
            d_model,
            nhead,
            dropout=0.1,
    ):
        super().__init__()
        # 自注意力用于增强
        self.self_attn = Attention(d_model, nhead, proj_drop=dropout)
        # 交叉注意力用于图文交互
        self.cross_attn = Attention(d_model, nhead, proj_drop=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x, mem):
        q = k = v = self.norm1(x)
        # x为视觉特征
        x = x + self.self_attn(q, k, v)
        q = self.norm2(x)
        # mem为文本特征投影
        x = x + self.cross_attn(q, mem, mem)
        x = x + self.dropout(self.mlp(self.norm3(x)))
        return x


@MODELS.register_module()
class CLIPVisionTransformer(nn.Module):
    def __init__(self, input_resolution=224, patch_size=32, width=768, layers=12, heads=12, output_dim=512,
                 drop_path_rate=0.0,  pretrained=None, get_embeddings=False, debug=False,
                same_dim=True,  **kwargs):
        super().__init__()
        self.debug = debug
        self.patch_size = patch_size
        self.same_dim = same_dim
        self.pretrained = pretrained
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))

        if same_dim:
            self.positional_embedding = nn.Parameter(
                scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))

        self.spatial_size = input_resolution // patch_size

        self.ln_pre = LayerNorm(width)
        self.get_embeddings = get_embeddings

        self.transformer = Transformer(width, layers, heads, drop_path_rate=drop_path_rate)


        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        embed_dim = width

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()

            state_dict = {}

            for k in checkpoint.keys():
                if k.startswith('visual.'):
                    new_k = k.replace('visual.', '')
                    state_dict[new_k] = checkpoint[k]

            if 'positional_embedding' in state_dict.keys():
                if self.positional_embedding.shape != state_dict['positional_embedding'].shape:
                    print(
                        f'Resize the pos_embed shape from {state_dict["positional_embedding"].shape} to {self.positional_embedding.shape}')
                    cls_pos = state_dict["positional_embedding"][0:1, :]

                    if self.patch_size == 32:
                        spatial_pos = F.interpolate(
                            state_dict["positional_embedding"][1:, ].reshape(1, 7, 7, 768).permute(0, 3, 1, 2),
                            size=(self.spatial_size, self.spatial_size), mode='bilinear')
                    elif self.patch_size == 16:
                        spatial_pos = F.interpolate(
                            state_dict["positional_embedding"][1:, ].reshape(1, 14, 14, 768).permute(0, 3, 1, 2),
                            size=(self.spatial_size, self.spatial_size), mode='bilinear')
                    spatial_pos = spatial_pos.reshape(768, self.spatial_size * self.spatial_size).permute(1, 0)


                    positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)
                    state_dict['positional_embedding'] = positional_embedding
                    assert self.positional_embedding.shape == state_dict['positional_embedding'].shape

            u, w = self.load_state_dict(state_dict, False)
            print(u, w, 'are misaligned params in vision transformer')

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        B, C, H, W = x.shape
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]

        pos = self.positional_embedding.to(x.dtype)
        cls_pos = pos[0, :] + self.class_embedding.to(x.dtype)

        spatial_pos = F.interpolate(
            pos[1:, ].reshape(1, self.spatial_size, self.spatial_size, C).permute(0, 3, 1, 2), size=(H, W),
            mode='bilinear')

        spatial_pos = spatial_pos.reshape(1, C, H * W).permute(0, 2, 1)
        pos = torch.cat([cls_pos.reshape(1, 1, C), spatial_pos], dim=1)
        x = x + pos
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND

        features = []
        for i, blk in enumerate(self.transformer.resblocks):
            x = blk(x)



        x = x.permute(1, 0, 2)
        x = self.ln_post(x)
        x = x @ self.proj

        global_embedding = x[:, 0]
        visual_embedding = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2)  # B C H W

        features.append([global_embedding, visual_embedding])

        return tuple(features)
# class CLIPVisionTransformer(nn.Module):
#     def __init__(self, input_resolution=256, patch_size=32, width=768, layers=12, heads=12, output_dim=512,
#                  drop_path_rate=0.0, pretrained=None, get_embeddings=True,
#                  same_dim=True, **kwargs):
#         super().__init__()
#         self.width = width
#         self.patch_size = patch_size
#         self.same_dim = same_dim
#         self.pretrained = pretrained
#         self.input_resolution = input_resolution
#         self.output_dim = output_dim
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
#         scale = width ** -0.5
#         self.class_embedding = nn.Parameter(scale * torch.randn(width))
#         self.spatial_size = input_resolution // patch_size

#         self.positional_embedding = nn.Parameter(scale * torch.randn((self.spatial_size ** 2 + 1, width)))

#         self.ln_pre = LayerNorm(width)
#         self.get_embeddings = get_embeddings

#         self.transformer = Transformer(width, layers, heads, drop_path_rate=drop_path_rate)

#         if get_embeddings:
#             self.ln_post = LayerNorm(width)
#             self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

#     def init_weights(self, pretrained=None):
#         pretrained = pretrained or self.pretrained
#         if isinstance(pretrained, str):
#             checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()
#             state_dict = {}
#             for k in checkpoint.keys():
#                 if k.startswith('visual.'):
#                     state_dict[k.replace('visual.', '')] = checkpoint[k]

#             # resize positional embedding if needed
#             if 'positional_embedding' in state_dict:
#                 if self.positional_embedding.shape != state_dict['positional_embedding'].shape:
#                     print(
#                         f"Resize pos_embed from {state_dict['positional_embedding'].shape} to {self.positional_embedding.shape}")
#                     cls_pos = state_dict["positional_embedding"][0:1, :]
#                     spatial_pos = state_dict["positional_embedding"][1:, :].reshape(1, self.width, int((state_dict[
#                                                                                                             "positional_embedding"].shape[
#                                                                                                             0] - 1) ** 0.5),
#                                                                                     -1)
#                     spatial_pos = F.interpolate(spatial_pos, size=(
#                     self.input_resolution // self.patch_size, self.input_resolution // self.patch_size),
#                                                 mode='bicubic')
#                     spatial_pos = spatial_pos.reshape(-1, self.width)
#                     state_dict['positional_embedding'] = torch.cat([cls_pos, spatial_pos], dim=0)

#             self.load_state_dict(state_dict, strict=False)
#     def forward(self, x: torch.Tensor):
#         x = self.conv1(x)  # shape = [*, width, grid, grid]
#         B, C, H, W = x.shape
#         x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
#         x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
#         x = torch.cat(
#             [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
#              x], dim=1)  # shape = [*, grid ** 2 + 1, width]

#         pos = self.positional_embedding.to(x.dtype)
#         cls_pos = pos[0, :] + self.class_embedding.to(x.dtype)
#         if self.same_dim:
#             spatial_pos = F.interpolate(
#                 pos[1:, ].reshape(1, self.spatial_size, self.spatial_size, C).permute(0, 3, 1, 2), size=(H, W),
#                 mode='bicubic')
#         else:
#             spatial_pos = F.interpolate(
#                 pos[1:, ].reshape(1, self.spatial_size, self.spatial_size_short, C).permute(0, 3, 1, 2), size=(H, W),
#                 mode='bilinear')
#         spatial_pos = spatial_pos.reshape(1, C, H * W).permute(0, 2, 1)
#         pos = torch.cat([cls_pos.reshape(1, 1, C), spatial_pos], dim=1)
#         x = x + pos
#         x = self.ln_pre(x)
#         x = x.permute(1, 0, 2)  # NLD -> LND

#         features = []
#         for i, blk in enumerate(self.transformer.resblocks):
#             x = blk(x)
#         if self.get_embeddings:
#             x = x.permute(1, 0, 2)
#             x = self.ln_post(x)
#             x = x @ self.proj

#             global_embedding = x[:, 0]
#             visual_embedding = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2)  # B C H W

#             features.append([global_embedding, visual_embedding])

#         return tuple(features)
# class CLIPVisionTransformer(nn.Module):
#     def __init__(self, input_resolution=224, patch_size=32, width=768, layers=12, heads=12, output_dim=512,
#                  drop_path_rate=0.0, out_indices=[3, 5, 7, 11], pretrained=None, get_embeddings=False, debug=False,
#                  use_fpn=True, same_dim=True, short_dim=192, **kwargs):
#         super().__init__()
#         self.debug = debug
#         self.patch_size = patch_size
#         self.same_dim = same_dim
#         self.use_fpn = use_fpn
#         self.pretrained = pretrained
#         self.input_resolution = input_resolution
#         self.output_dim = output_dim
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

#         scale = width ** -0.5
#         self.class_embedding = nn.Parameter(scale * torch.randn(width))

#         if same_dim:
#             self.positional_embedding = nn.Parameter(
#                 scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
#         else:
#             self.positional_embedding = nn.Parameter(
#                 scale * torch.randn((input_resolution // patch_size) * (short_dim // patch_size) + 1, width))
#         self.spatial_size = input_resolution // patch_size
#         if not same_dim:
#             self.spatial_size_short = short_dim // patch_size
#         self.ln_pre = LayerNorm(width)
#         self.get_embeddings = get_embeddings

#         self.transformer = Transformer(width, layers, heads, drop_path_rate=drop_path_rate)

#         self.out_indices = out_indices

#         if get_embeddings:
#             self.ln_post = LayerNorm(width)
#             self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

#         embed_dim = width

#         if use_fpn:
#             if patch_size == 16:
#                 self.fpn1 = nn.Sequential(
#                     nn.GroupNorm(1, embed_dim),
#                     nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
#                     nn.SyncBatchNorm(embed_dim),
#                     nn.GELU(),
#                     nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
#                 )

#                 self.fpn2 = nn.Sequential(
#                     nn.GroupNorm(1, embed_dim),
#                     nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
#                 )

#                 self.fpn3 = nn.GroupNorm(1, embed_dim)

#                 self.fpn4 = nn.Sequential(
#                     nn.GroupNorm(1, embed_dim),
#                     nn.MaxPool2d(kernel_size=2, stride=2)
#                 )

#             elif patch_size == 8:
#                 self.fpn1 = nn.Sequential(
#                     nn.GroupNorm(1, embed_dim),
#                     nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
#                 )

#                 self.fpn2 = nn.GroupNorm(1, embed_dim)

#                 self.fpn3 = nn.Sequential(
#                     nn.GroupNorm(1, embed_dim),
#                     nn.MaxPool2d(kernel_size=2, stride=2),
#                 )

#                 self.fpn4 = nn.Sequential(
#                     nn.GroupNorm(1, embed_dim),
#                     nn.MaxPool2d(kernel_size=4, stride=4),
#                 )

#     def init_weights(self, pretrained=None):
#         pretrained = pretrained or self.pretrained
#         if isinstance(pretrained, str):
#             checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()

#             state_dict = {}

#             for k in checkpoint.keys():
#                 if k.startswith('visual.'):
#                     new_k = k.replace('visual.', '')
#                     state_dict[new_k] = checkpoint[k]

#             if 'positional_embedding' in state_dict.keys():
#                 if self.positional_embedding.shape != state_dict['positional_embedding'].shape:
#                     print(
#                         f'Resize the pos_embed shape from {state_dict["positional_embedding"].shape} to {self.positional_embedding.shape}')
#                     cls_pos = state_dict["positional_embedding"][0:1, :]
#                     if self.same_dim:
#                         if self.patch_size == 32:
#                             spatial_pos = F.interpolate(
#                                 state_dict["positional_embedding"][1:, ].reshape(1, 7, 7, 768).permute(0, 3, 1, 2),
#                                 size=(self.spatial_size, self.spatial_size), mode='bilinear')
#                         elif self.patch_size == 16:
#                             spatial_pos = F.interpolate(
#                                 state_dict["positional_embedding"][1:, ].reshape(1, 14, 14, 768).permute(0, 3, 1, 2),
#                                 size=(self.spatial_size, self.spatial_size), mode='bilinear')
#                         spatial_pos = spatial_pos.reshape(768, self.spatial_size * self.spatial_size).permute(1, 0)
#                     else:
#                         if self.patch_size == 32:
#                             spatial_pos = F.interpolate(
#                                 state_dict["positional_embedding"][1:, ].reshape(1, 7, 7, 768).permute(0, 3, 1, 2),
#                                 size=(self.spatial_size, self.spatial_size_short), mode='bilinear')
#                         elif self.patch_size == 16:
#                             spatial_pos = F.interpolate(
#                                 state_dict["positional_embedding"][1:, ].reshape(1, 14, 14, 768).permute(0, 3, 1, 2),
#                                 size=(self.spatial_size, self.spatial_size_short), mode='bilinear')
#                         spatial_pos = spatial_pos.reshape(768, self.spatial_size * self.spatial_size_short).permute(1,
#                                                                                                                     0)
#                     positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)
#                     state_dict['positional_embedding'] = positional_embedding
#                     assert self.positional_embedding.shape == state_dict['positional_embedding'].shape

#             u, w = self.load_state_dict(state_dict, False)
#             print(u, w, 'are misaligned params in vision transformer')

#     def forward(self, x: torch.Tensor):
#         x = self.conv1(x)  # shape = [*, width, grid, grid]
#         B, C, H, W = x.shape
#         x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
#         x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
#         x = torch.cat(
#             [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
#              x], dim=1)  # shape = [*, grid ** 2 + 1, width]

#         pos = self.positional_embedding.to(x.dtype)
#         cls_pos = pos[0, :] + self.class_embedding.to(x.dtype)
#         if self.same_dim:
#             spatial_pos = F.interpolate(
#                 pos[1:, ].reshape(1, self.spatial_size, self.spatial_size, C).permute(0, 3, 1, 2), size=(H, W),
#                 mode='bilinear')
#         else:
#             spatial_pos = F.interpolate(
#                 pos[1:, ].reshape(1, self.spatial_size, self.spatial_size_short, C).permute(0, 3, 1, 2), size=(H, W),
#                 mode='bilinear')
#         spatial_pos = spatial_pos.reshape(1, C, H * W).permute(0, 2, 1)
#         pos = torch.cat([cls_pos.reshape(1, 1, C), spatial_pos], dim=1)
#         x = x + pos
#         x = self.ln_pre(x)
#         x = x.permute(1, 0, 2)  # NLD -> LND

#         features = []
#         for i, blk in enumerate(self.transformer.resblocks):
#             x = blk(x)
#             if i in self.out_indices:
#                 xp = x.permute(1, 0, 2)[:, 1:, :].permute(0, 2, 1).reshape(B, -1, H, W)
#                 features.append(xp.contiguous())

#         if self.use_fpn:
#             ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
#             for i in range(len(features)):
#                 features[i] = ops[i](features[i])

#         if self.get_embeddings:
#             x = x.permute(1, 0, 2)
#             x = self.ln_post(x)
#             x = x @ self.proj

#             global_embedding = x[:, 0]
#             visual_embedding = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2)  # B C H W

#             features.append([global_embedding, visual_embedding])

#         return tuple(features)
@MODELS.register_module()
class CLIPprompt(nn.Module):
    def __init__(self, 
                 input_resolution=224, 
                 patch_size=32, 
                 width=768, 
                 layers=12, 
                 heads=12, 
                 output_dim=1024,
                 drop_path_rate=0.0, 
                 pretrained=None, 
                 get_embeddings=False, 
                 debug=False,
                 same_dim=True, 
                 prompt_config=dict(deep=True, num_tokens=17),
                 transformer=None,  # 传入 CLIP Transformer
                 **kwargs):
        super().__init__()
        self.debug = debug
        self.patch_size = patch_size
        self.same_dim = same_dim
        self.pretrained = pretrained
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.get_embeddings = get_embeddings
        self.prompt_config = prompt_config
        self.num_tokens = prompt_config['num_tokens']

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.spatial_size = input_resolution // patch_size

        if same_dim:
            # Patch positional embedding, 不包含 cls token
            self.positional_embedding = nn.Parameter(
                scale * torch.randn((self.spatial_size ** 2), width)
            )

        # LayerNorm
        self.ln_pre = nn.LayerNorm(width)
        self.ln_post = nn.LayerNorm(width)

        # Transformer
        self.transformer = Transformer(width, layers, heads, drop_path_rate=drop_path_rate)
        # self.resblocks = transformer.resblocks

        # Prompt token
        self.prompt_embed = nn.Parameter(torch.zeros(1, self.num_tokens, width))
        nn.init.trunc_normal_(self.prompt_embed, std=0.02)

        # Deep prompt token
        if self.prompt_config.get('deep', False):
            self.deep_prompt_embed = nn.Parameter(torch.zeros(layers - 1, self.num_tokens, width))
            nn.init.trunc_normal_(self.deep_prompt_embed, std=0.02)

        # 输出投影 (和原始 CLIPVisionTransformer 对齐)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        # 冻结 backbone
        self._freeze_backbone()

    def _freeze_backbone(self):
        """冻结 patch embedding、位置编码、transformer、LayerNorm"""
        for param in self.conv1.parameters():
            param.requires_grad = False
        self.positional_embedding.requires_grad = False

        for blk in self.transformer.resblocks:
            for param in blk.parameters():
                param.requires_grad = False

        for param in self.ln_pre.parameters():
            param.requires_grad = False
        for param in self.ln_post.parameters():
            param.requires_grad = False

        # 保证 prompt token 可训练
        self.prompt_embed.requires_grad = True
        if hasattr(self, 'deep_prompt_embed'):
            self.deep_prompt_embed.requires_grad = True

    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        x = self.conv1(x)  # B, C, H, W
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1).permute(0, 2, 1)  # B, N, C

        # Add positional embedding to patch tokens
        pos = self.positional_embedding.to(x.dtype)  # N, C
        pos = pos.unsqueeze(0).expand(B, -1, -1)    # B, N, C
        x = x + pos

        # LayerNorm before transformer
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND for transformer

        # Add shallow prompt
        prompt_tokens = self.prompt_embed.expand(B, -1, -1).permute(1, 0, 2)  # L_prompt, B, D
        x_ = torch.cat([prompt_tokens, x], dim=0)

        # Forward through transformer
        for i, blk in enumerate(self.transformer.resblocks):
            if self.prompt_config.get('deep', False) and i > 0:
                deep_prompt = self.deep_prompt_embed[i - 1].expand(B, -1, -1).permute(1, 0, 2)
                x_ = torch.cat([deep_prompt, x_], dim=0)
            x_ = blk(x_)
            # Remove prompt tokens for next layer (except last layer)
            if i < len(self.transformer.resblocks) - 1:
                x_ = x_[self.num_tokens:, :, :]

        # x_ shape: L, B, D -> B, L, D
        x_ = x_.permute(1, 0, 2)
        
        # 分割 prompt token 和 patch token
        prompt_tokens_out = self.ln_post(x_[:, :self.num_tokens, :])  # B, K, D
        patch_tokens_out = self.ln_post(x_[:, self.num_tokens:, :])   # B, N_patch, D
        prompt_tokens_out = prompt_tokens_out @ self.proj
        return prompt_tokens_out

@MODELS.register_module()
class CLIPTextencoder(nn.Module):
    def __init__(self, context_length=77,
                 vocab_size=49408,
                 transformer_width=512,
                 transformer_heads=8,
                 transformer_layers=12,
                 embed_dim=1024,
                 out_dim=256,
                 pretrained=None, **kwargs):
        super().__init__()

        self.pretrained = pretrained

        self.context_length = context_length

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()

            state_dict = {}

            for k in checkpoint.keys():
                if k.startswith('transformer.'):
                    state_dict[k] = checkpoint[k]

                if k == 'positional_embedding' or k == 'text_projection' or k.startswith(
                        'token_embedding') or k.startswith('ln_final'):
                    if k == 'positional_embedding' and checkpoint[k].size(0) > self.context_length:
                        checkpoint[k] = checkpoint[k][:self.context_length]
                        print('positional_embedding is tuncated from 77 to', self.context_length)
                    state_dict[k] = checkpoint[k]

            u, w = self.load_state_dict(state_dict, False)
            print(u, w, 'are misaligned params in text encoder')

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x
class PromptInserter(nn.Module):
    def __init__(self, ctx_lens: int, embed_dim: int):
        """
        total_ctx_len: int, 可学习 token 总长度
        embed_dim: int, token embedding 维度
        """
        super().__init__()
        self.total_ctx_len = ctx_lens
        self.embed_dim = embed_dim

        # learnable tokens 池
        self.ctx_tokens = nn.Parameter(torch.randn(ctx_lens, embed_dim) * 0.02)
        nn.init.trunc_normal_(self.ctx_tokens, std=0.02)
    def forward(self, embedded_tokens: torch.Tensor) -> torch.Tensor:
        """
        embedded_tokens: [B, L, D]
        tokenized_ids: [B, L]
        """
        B, L, D = embedded_tokens.shape

        # 1. 随机划分 ctx tokens 的长度（至少 1 个 token）
        remaining = self.total_ctx_len
        segments = []
        while remaining > 0:
            seg_len = torch.randint(1, remaining + 1, (1,)).item()
            segments.append(seg_len)
            remaining -= seg_len

        # 2. 根据 segments 切 ctx_tokens，并 batch 复制
        ctx_slices = []
        start_idx = 0
        for seg_len in segments:
            ctx_slice = self.ctx_tokens[start_idx:start_idx + seg_len].unsqueeze(0).expand(B, -1, -1)
            ctx_slices.append(ctx_slice)
            start_idx += seg_len

        # 3. 拆分原 embedding，避开 CLS token
        x0 = embedded_tokens[:, :1, :]  # CLS token
        x1 = embedded_tokens[:, 1:, :]  # class tokens

        parts = [x0]
        for ctx in ctx_slices:
            # 随机插入位置
            pos = torch.randint(0, x1.shape[1] + 1, (1,)).item()
            parts.append(x1[:, :pos, :])
            parts.append(ctx)
            x1 = x1[:, pos:, :]
        parts.append(x1)

        # 4. 拼接所有
        new_tokens = torch.cat(parts, dim=1)  # [B, L+total_ctx_len, D]
        return new_tokens
# @BACKBONES.register_module()
# class CLIPTextContextEncoder(nn.Module):
#     def __init__(self, context_length=24,
#                  vocab_size=49408,
#                  transformer_width=512,
#                  transformer_heads=8,
#                  transformer_layers=12,
#                  embed_dim=1024,
#                  out_dim=256,
#                  vitae=False,
#                  pretrained=None, **kwargs):
#         super().__init__()

#         self.pretrained = pretrained
#         self.vitae = vitae
#         self.transformer_width = transformer_width
#         self.context_length = context_length

#         self.transformer = Transformer(
#             width=transformer_width,
#             layers=transformer_layers,
#             heads=transformer_heads,
#             attn_mask=self.build_attention_mask()
#         )

#         self.embed_dim = embed_dim

#         self.vocab_size = vocab_size
#         self.token_embedding = nn.Embedding(vocab_size, transformer_width)
#         self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
#         self.ln_final = LayerNorm(transformer_width)
#         self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
#         self.prompt_inserter = PromptInserter(ctx_lens=5, embed_dim=transformer_width)

#     def init_weights(self, pretrained=None):
#         pretrained = pretrained or self.pretrained
#         if isinstance(pretrained, str):
#             if self.vitae:
#                 checkpoint = torch.load(pretrained, map_location='cpu')['state_dict']
#             else:
#                 checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()

#             state_dict = {}

#             for k in checkpoint.keys():
#                 if k.startswith('module.'):  # For ViTAE loading
#                     new_k = k.replace('module.', '')
#                     if new_k.startswith('transformer.'):
#                         state_dict[new_k] = checkpoint[k]

#                     if new_k == 'positional_embedding' or new_k == 'text_projection' or new_k.startswith(
#                             'token_embedding') or new_k.startswith('ln_final'):
#                         if new_k == 'positional_embedding' and checkpoint[k].size(0) > self.context_length:
#                             checkpoint[k] = checkpoint[k][:self.context_length]
#                             print('positional_embedding is tuncated from 77 to', self.context_length)
#                         state_dict[new_k] = checkpoint[k]
#                 else:
#                     if k.startswith('transformer.'):
#                         state_dict[k] = checkpoint[k]

#                     if k == 'positional_embedding' or k == 'text_projection' or k.startswith(
#                             'token_embedding') or k.startswith('ln_final'):
#                         if k == 'positional_embedding' and checkpoint[k].size(0) > self.context_length:
#                             checkpoint[k] = checkpoint[k][:self.context_length]
#                             print('positional_embedding is tuncated from 77 to', self.context_length)
#                         state_dict[k] = checkpoint[k]

#             u, w = self.load_state_dict(state_dict, False)
#             print(u, w, 'are misaligned params in text encoder')

#     def build_attention_mask(self):
#         # lazily create causal attention mask, with full attention between the vision tokens
#         # pytorch uses additive attention mask; fill with -inf
#         mask = torch.empty(self.context_length, self.context_length)
#         mask.fill_(float("-inf"))
#         mask.triu_(1)  # zero out the lower diagonal
#         return mask

#     def forward(self, text, context=None):
#         seq_len = min(text.shape[1], self.context_length)
#         text = text[:, :seq_len]  # 截断多余 token
#         x_text = self.token_embedding(text)  # n_clas, n_text, C
#         new_tokens = self.prompt_inserter(x_text)
#         if context is not None:
#             K, N1, C = x_text.shape
#             B, N2, C = context.shape

#             eos_indx = text.argmax(dim=-1) + N2
#             eos_indx = eos_indx.reshape(1, K).expand(B, K).reshape(-1)

#             x_text = x_text.reshape(1, K, N1, C).expand(B, K, N1, C)
#             context = context.reshape(B, 1, N2, C).expand(B, K, N2, C)

#             x = torch.cat([x_text[:, :, 0:1], context, x_text[:, :, 1:]], dim=2).reshape(B * K, N1 + N2, C)
#         else:
#             # 直接把 text reshape 成 batch
#             B = 1
#             K, N1, C = x_text.shape
#             x = x_text.reshape(B * K, N1, C)
#             eos_indx = None
#         # x = x + self.positional_embedding
#         x = x + self.positional_embedding[:x.shape[1], :]
#         x = x.permute(1, 0, 2)  # NLD -> LND
#         x = self.transformer(x)
#         x = x.permute(1, 0, 2)  # LND -> NLD
#         x = self.ln_final(x)
#         if self.transformer_width != self.embed_dim:
#             x = x @ self.text_projection
#         if eos_indx is not None:
#             x = x[torch.arange(x.shape[0]), eos_indx] @ self.text_projection
#             x = x.reshape(B, K, self.embed_dim)
#         return x
@BACKBONES.register_module()
class CLIPTextContextEncoder(nn.Module):
    def __init__(self, context_length=24,
                 vocab_size=49408,
                 transformer_width=512,
                 transformer_heads=8,
                 transformer_layers=12,
                 embed_dim=1024,
                 out_dim=1024,
                 vitae=False,
                 pretrained=None, **kwargs):
        super().__init__()

        self.pretrained = pretrained
        self.vitae = vitae
        self.transformer_width = transformer_width
        self.context_length = context_length
        
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.embed_dim = embed_dim

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            if self.vitae:
                checkpoint = torch.load(pretrained, map_location='cpu')['state_dict']
            else:
                checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()

            state_dict = {}

            for k in checkpoint.keys():
                if k.startswith('module.'):  # For ViTAE loading
                    new_k = k.replace('module.', '')
                    if new_k.startswith('transformer.'):
                        state_dict[new_k] = checkpoint[k]

                    if new_k == 'positional_embedding' or new_k == 'text_projection' or new_k.startswith(
                            'token_embedding') or new_k.startswith('ln_final'):
                        if new_k == 'positional_embedding' and checkpoint[k].size(0) > self.context_length:
                            checkpoint[k] = checkpoint[k][:self.context_length]
                            print('positional_embedding is tuncated from 77 to', self.context_length)
                        state_dict[new_k] = checkpoint[k]
                else:
                    if k.startswith('transformer.'):
                        state_dict[k] = checkpoint[k]

                    if k == 'positional_embedding' or k == 'text_projection' or k.startswith(
                            'token_embedding') or k.startswith('ln_final'):
                        if k == 'positional_embedding' and checkpoint[k].size(0) > self.context_length:
                            checkpoint[k] = checkpoint[k][:self.context_length]
                            print('positional_embedding is tuncated from 77 to', self.context_length)
                        state_dict[k] = checkpoint[k]

            u, w = self.load_state_dict(state_dict, False)
            print(u, w, 'are misaligned params in text encoder')

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, text, context=None):
        """
        text: [B_text, seq_len]
        context: 可选外部 context 特征, [B, N_context, C]
        """
        # 截断多余 token
        seq_len = min(text.shape[1], self.context_length)
        text = text[:, :seq_len]  
    
        # token embedding
        x_text = self.token_embedding(text)  # [B_text, N_text, C]
        B_text, N_text, C = x_text.shape
    
        # if context is not None:
        #     B, N_context, C_ctx = context.shape
        #     assert C_ctx == C, "context embedding dimension mismatch"
    
        #     # 动态 expand
        #     x_text = x_text.unsqueeze(0).expand(B, -1, -1, -1) if x_text.dim() == 3 else x_text
        #     context = context.unsqueeze(1).expand(-1, B_text, -1, -1)  # [B, B_text, N_context, C]
    
        #     # 拼接 text + context
        #     x_text = x_text.permute(1, 0, 2, 3)  # [B_text, B, N_text, C]
        #     x = torch.cat([x_text[:, :, 0:1], context, x_text[:, :, 1:]], dim=2)  # [B_text, B, N_text+N_context, C]
        #     B_total, _, _ = x.shape[1:]  # 注意 batch 维度
        #     x = x.reshape(B * B_text, N_text + N_context, C)
        if context is not None:
            B, N_context, C_ctx = context.shape
            assert C_ctx == C, "context embedding dimension mismatch"
        
            # 保持一致的batch维度顺序
            x_text = x_text.unsqueeze(0).expand(B, -1, -1, -1) if x_text.dim() == 3 else x_text
            x_text = x_text.permute(1, 0, 2, 3)  # [B_text, B, N_text, C]
        
            context = context.unsqueeze(0).expand(B_text, -1, -1, -1)  # [B_text, B, N_context, C]
        
            # 拼接
            x = torch.cat([x_text[:, :, 0:1], context, x_text[:, :, 1:]], dim=2)  # [B_text, B, N_text+N_context, C]
            x = x.reshape(B * B_text, N_text + N_context, C)

        else:
            # 没有 context
            B = x_text.shape[0]
            x = x_text  # [B, N_text, C]
        
        # 添加位置编码
        x = x + self.positional_embedding[:x.shape[1], :]
        
        # transformer
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
    
        # 投影
        if self.transformer_width != self.embed_dim:
            x = x @ self.text_projection
    
        # 处理 eos token
        if context is not None:
            # 这里直接取拼接后的 eos token
            eos_indx = text.argmax(dim=-1)
            x = x[torch.arange(x.shape[0]), eos_indx] @ self.text_projection
            x = x.reshape(B, B_text, self.embed_dim)
        
        return x


@MODELS.register_module()
class PromptEncoderWithoutPositionemb(nn.Module):
    def __init__(self, prompt_num=17,
                 transformer_width=512,
                 transformer_heads=8,
                 transformer_layers=1,
                 embed_dim=1024,
                 pretrained=None, **kwargs):
        super().__init__()

        self.pretrained = pretrained

        self.prompt_num = prompt_num

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=None
        )

        self.embed_dim = embed_dim

        self.positional_embedding = nn.Parameter(torch.empty(self.prompt_num, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

    def init_weights(self, pretrained=None):
        return None

    def forward(self, prompt_emb):
        B, K, C = prompt_emb.shape

        x = prompt_emb
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        x = x.reshape(B, K, self.embed_dim)
        return x

from timm.models.layers import DropPath

# class ContextDecoder(nn.Module):
#     def __init__(self,
#                  transformer_width=256,
#                  transformer_heads=4,
#                  transformer_layers=6,
#                  visual_dim=1024,
#                  dropout=0.1,
#                  **kwargs):
#         super().__init__()

#         self.memory_proj = nn.Sequential(
#             nn.LayerNorm(visual_dim),
#             nn.Linear(visual_dim, transformer_width),
#             nn.LayerNorm(transformer_width),
#         )

#         self.text_proj = nn.Sequential(
#             nn.LayerNorm(visual_dim),
#             nn.Linear(visual_dim, transformer_width),
#         )

#         self.decoder = nn.ModuleList([
#             TransformerDecoderLayer(transformer_width, transformer_heads, dropout) for _ in range(transformer_layers)
#         ])

#         self.out_proj = nn.Sequential(
#             nn.LayerNorm(transformer_width),
#             nn.Linear(transformer_width, visual_dim)
#         )

#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)

#     def forward(self, text, visual):
#         B, N, C = visual.shape
#         visual = self.memory_proj(visual)
#         x = self.text_proj(text)

#         for layer in self.decoder:
#             x = layer(x, visual)

#         return self.out_proj(x)

class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        return self.gamma * x
@MODELS.register_module()
class ContextDecoder(nn.Module):
    def __init__(self,
                 transformer_width=256,
                 transformer_heads=4,
                 transformer_layers=6,  # 减少层数
                 visual_dim=1024,
                 dropout=0.1,**kwargs):
        super().__init__()

        # 合并 proj 层
        self.proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, transformer_width),
        )

        # 使用轻量 TransformerDecoderLayer
        self.decoder = nn.ModuleList([
            LiteTransformerDecoderLayer(transformer_width, transformer_heads, dropout)
            for _ in range(transformer_layers)
        ])

        # 输出层
        self.out_proj = nn.Sequential(
            nn.LayerNorm(transformer_width),
            nn.Linear(transformer_width, visual_dim)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, text, visual):
        # text, visual: (B, N, C)
        # visual = self.proj(visual)
        # text = self.proj(text)

        for layer in self.decoder:
            text = layer(text, visual)

        # return self.out_proj(text)
        return text

# class CLIPTextContextEncoder1(nn.Module):
#     def __init__(self,
#                  vocab_size=49408,
#                  transformer_width=512,
#                  transformer_heads=8,
#                  transformer_layers=12,
#                  embed_dim=1024,
#                  text_len=15,
#                  learnable_len=5,       # 总隐式 token 数
#                  num_keypoints=17,      # 关键点数量，用于 per-kpt tokens
#                  num_segments=3,        # 分段插入数量（插入点数）
#                  pretrained=None, **kwargs):
#         super().__init__()
#         self.pretrained = pretrained
#         self.transformer_width = transformer_width
#         self.text_len = text_len
#         self.learnable_len = learnable_len
#         self.num_segments = num_segments
#         self.num_keypoints = num_keypoints
#         self.total_len = text_len + learnable_len

#         self.positional_embedding = nn.Parameter(
#             torch.empty(self.total_len, transformer_width)
#         )
#         self.transformer = Transformer(
#             width=transformer_width,
#             layers=transformer_layers,
#             heads=transformer_heads,
#             attn_mask=None  # ⚠️ 不要固定 mask
#         )
#         self.token_embedding = nn.Embedding(vocab_size, transformer_width)

#         # ★ 优化1：per-keypoint learnable tokens
#         # 每个关键点有自己专属的隐式先验，不再全局共享
#         # shape: (num_keypoints, learnable_len, transformer_width)
#         self.learnable_tokens = nn.Parameter(
#             torch.randn(num_keypoints, learnable_len, transformer_width) * 0.02
#         )

#         # ★ 优化2：可学习的分段权重
#         # 控制 learnable_len 个 token 如何分配到 num_segments 个插入点
#         # 训练时学习最优分配，而非随机
#         self.segment_logits = nn.Parameter(
#             torch.zeros(learnable_len, num_segments)
#         )  # softmax 后得到每个 token 属于哪个 segment 的软分配

#         self.ln_final = LayerNorm(transformer_width)
#         self.text_projection = nn.Parameter(
#             torch.empty(transformer_width, embed_dim)
#         )
#         nn.init.trunc_normal_(self.positional_embedding, std=0.01)
#         nn.init.trunc_normal_(self.text_projection, std=0.02)

#     def build_attention_mask(self, seq_len, device):
#         mask = torch.full(
#             (seq_len, seq_len),
#             float('-inf'),
#             device=device
#         )
#         mask.triu_(1)
#         return mask

#     # def _split_tokens_into_segments(self, learnable, training=True):
#     #     """
#     #     将 learnable_len 个 token 分配到 num_segments 个段
#     #     training=True: 用 Gumbel-softmax 做可微分的软分配
#     #     training=False: 用 argmax 做硬分配

#     #     learnable: (B, learnable_len, C)
#     #     return: list of (B, n_i, C), len = num_segments, sum(n_i) = learnable_len
#     #     """
#     #     if training:
#     #         # Gumbel-softmax：可微分，训练时学习最优分配
#     #         # segment_logits: (learnable_len, num_segments)
#     #         assign = F.gumbel_softmax(
#     #             self.segment_logits, tau=1.0, hard=False, dim=-1
#     #         )  # (learnable_len, num_segments) 软权重
#     #     else:
#     #         # 推理时硬分配
#     #         idx = self.segment_logits.argmax(dim=-1)  # (learnable_len,)
#     #         assign = F.one_hot(idx, self.num_segments).float()

#     #     # 按 segment 聚合 token
#     #     # assign: (learnable_len, num_segments)
#     #     # learnable: (B, learnable_len, C)
#     #     segments = []
#     #     for s in range(self.num_segments):
#     #         w = assign[:, s]                              # (learnable_len,)
#     #         # 软加权求和（training）或硬 mask（inference）
#     #         seg = (learnable * w.unsqueeze(0).unsqueeze(-1)).sum(dim=1, keepdim=True)
#     #         # 实际上推理时是 hard mask，这里统一用加权和简化实现
#     #         # 真正的 hard 版本：seg_tokens = learnable[:, w.bool(), :]
#     #         segments.append(seg)                         # (B, 1, C) 软版本
#     #     return segments

#     def _get_insert_positions(self, L, training=True):
#         """
#         确定 num_segments 个插入位置
#         训练：随机采样（每个样本独立）
#         推理：均匀分布
#         返回升序位置列表，长度 = num_segments
#         """
#         if training:
#             # 从 [1, L-1] 中采样 num_segments 个不重复位置
#             positions = sorted(
#                 torch.randperm(L - 1)[:self.num_segments].tolist()
#             )
#             positions = [p + 1 for p in positions]      # 避开位置0（通常是 SOS）
#         else:
#             step = L // (self.num_segments + 1)
#             positions = [step * (i + 1) for i in range(self.num_segments)]
#         return positions

#     def _split_tokens_into_segments(self, learnable, training=True):
#         """
#         learnable: (B, learnable_len, C)
    
#         return:
#             segments: list of tensors
#                 each tensor: (B, n_i, C)
#                 保证：
#                     1. batch 对齐
#                     2. 所有 token 被分配
#                     3. 不出现 None（避免后续 cat 出错）
#         """
#         B, N, C = learnable.shape
#         S = self.num_segments
    
#         # -------------------------------
#         # 1. batch-aware assignment
#         # -------------------------------
#         if training:
#             # (B, N, S)
#             logits = self.segment_logits.unsqueeze(0).expand(B, -1, -1)
    
#             assign = F.gumbel_softmax(
#                 logits, tau=1.0, hard=True, dim=-1
#             )  # one-hot per token
#         else:
#             idx = self.segment_logits.argmax(dim=-1)          # (N,)
#             one_hot = F.one_hot(idx, S).float()               # (N, S)
#             assign = one_hot.unsqueeze(0).expand(B, -1, -1)   # (B, N, S)
    
#         # -------------------------------
#         # 2. 构建 segments（关键修复点）
#         # -------------------------------
#         segments = []
    
#         for s in range(S):
#             mask = assign[:, :, s]  # (B, N)
    
#             seg_list = []
#             max_len = 0
    
#             # 每个 batch 单独选（避免 broadcast）
#             for b in range(B):
#                 m = mask[b].bool()
#                 selected = learnable[b, m, :]  # (n_b, C)
    
#                 seg_list.append(selected)
#                 max_len = max(max_len, selected.shape[0])
    
#             # -------------------------------
#             # 3. pad 对齐 batch（关键！！）
#             # -------------------------------
#             if max_len == 0:
#                 # 防止空 segment → 用一个 dummy token（不会影响训练）
#                 seg = torch.zeros(B, 1, C, device=learnable.device, dtype=learnable.dtype)
#             else:
#                 padded = []
#                 for selected in seg_list:
#                     if selected.shape[0] < max_len:
#                         pad_len = max_len - selected.shape[0]
#                         pad = torch.zeros(
#                             pad_len, C,
#                             device=learnable.device,
#                             dtype=learnable.dtype
#                         )
#                         selected = torch.cat([selected, pad], dim=0)
#                     padded.append(selected)
    
#                 seg = torch.stack(padded, dim=0)  # (B, max_len, C)
    
#             segments.append(seg)
    
#         return segments
    
    
#     def forward(self, text, keypoint_ids=None):
#         device = text.device
#         B = text.shape[0]                                  # 实际 B = batch * K
    
#         # ---- 1. text embedding ----
#         seq_len = min(text.shape[1], self.text_len)
#         text = text[:, :seq_len]
#         x_text = self.token_embedding(text)                # (B, L, C)
#         L, C = x_text.shape[1], x_text.shape[2]
    
#         # ---- 2. per-keypoint learnable tokens ----
#         if keypoint_ids is not None:
#             # keypoint_ids: (B,)，每个样本对应哪个关键点
#             learnable = self.learnable_tokens[keypoint_ids]    # (B, learnable_len, C)
#         else:
#             # 退化：用所有关键点的均值
#             learnable = self.learnable_tokens.mean(0, keepdim=True).expand(B, -1, -1)
#                                                                # (B, learnable_len, C)
    
#         # ---- 3. 分段 token（hard 分配，每段有实际 token） ----
#         segments = self._split_tokens_into_segments(
#             learnable, training=self.training
#         )  # list of (B, n_i, C) or None
    
#         positions = self._get_insert_positions(L, training=self.training)
    
#         # ---- 4. 插入（从后往前避免位置偏移） ----
#         x = x_text                                         # (B, L, C)
#         # valid_insertions = [
#         #     (pos, seg) for pos, seg in zip(positions, segments)
#         #     if seg is not None
#         # ]
#         valid_insertions = list(zip(positions, segments))
#         for pos, seg in reversed(valid_insertions):
#             # seg: (B, n_i, C)  x: (B, current_len, C)
#             # 确保 batch 维度一致
#             assert seg.shape[0] == x.shape[0], \
#                 f"batch mismatch: seg {seg.shape[0]} vs x {x.shape[0]}"
#             x = torch.cat([x[:, :pos, :], seg, x[:, pos:, :]], dim=1)
    
#         max_len = x.shape[1]
    
#         # ---- 5. positional embedding ----
#         pe_len = self.positional_embedding.shape[0]
#         if max_len <= pe_len:
#             pe = self.positional_embedding[:max_len, :]
#         else:
#             pad = torch.zeros(max_len - pe_len, C, device=device)
#             pe = torch.cat([self.positional_embedding, pad], dim=0)
#         x = x + pe.unsqueeze(0)                            # broadcast over B
    
#         # ---- 6. transformer ----
#         # x = x.permute(1, 0, 2)                             # (L', B, C)
#         # x = self.transformer(x)
#         # x = x.permute(1, 0, 2)                             # (B, L', C)
#         # ---- 6. transformer ----
#         seq_len = x.shape[1]
        
#         attn_mask = self.build_attention_mask(seq_len, x.device)
        
#         x = x.permute(1, 0, 2)  # (L, B, C)
        
#         # 👉 动态注入 mask（关键）
#         for block in self.transformer.resblocks:
#             block.attn_mask = attn_mask
        
#         x = self.transformer(x)
        
#         x = x.permute(1, 0, 2)
#         x = self.ln_final(x)
#         x = x @ self.text_projection                       # (B, L', embed_dim)
#         tokenp = x
    
#         # ---- 7. EOS feature ----
#         with torch.no_grad():
#             eos_pos = text.argmax(dim=-1).clamp(max=max_len - 1)  # (B,)
#         x = x[torch.arange(B, device=device), eos_pos]    # (B, embed_dim)
#         return x
#         # return tokenp, x
import torch
import torch.nn as nn
@MODELS.register_module()
# class CLIPTextContextEncoder1(nn.Module):
#     # def __init__(self, 
#     #              context_length=77,      # 标准 CLIP 为 77
#     #              vocab_size=49408,
#     #              transformer_width=512,
#     #              transformer_heads=8,
#     #              transformer_layers=12,
#     #              embed_dim=1024,
#     #              **kwargs):
#     def __init__(self, 
#                  context_length=24,
#                  vocab_size=49408,
#                  transformer_width=512,
#                  transformer_heads=8,
#                  transformer_layers=12,
#                  embed_dim=1024,
#                  out_dim=1024,
#                  text_len=15,
#                  # learnable_len=5,
#                  learnable_len=5,
#                  vitae=False,
#                  pretrained=None, **kwargs):
#         super().__init__()

#         self.context_length = context_length
#         self.transformer_width = transformer_width
#         self.vocab_size = vocab_size

#         # 1. 基础嵌入层
#         self.token_embedding = nn.Embedding(vocab_size, transformer_width)
#         self.positional_embedding = nn.Parameter(torch.empty(context_length, transformer_width))
        
#         # 2. Transformer 结构
#         self.transformer = Transformer(
#             width=transformer_width,
#             layers=transformer_layers,
#             heads=transformer_heads,
#             attn_mask=self.build_attention_mask()
#         )

#         # 3. 输出层
#         self.ln_final = LayerNorm(transformer_width)
#         self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

#         # 初始化
#         nn.init.trunc_normal_(self.positional_embedding, std=0.01)
#         nn.init.trunc_normal_(self.text_projection, std=0.02)
#         mask = torch.empty(context_length, context_length).fill_(float("-inf")).triu_(1)
#         self.register_buffer('causal_mask', mask)
#     def build_attention_mask(self):
#         # 标准因果遮罩 (Causal Mask)
#         mask = torch.empty(self.context_length, self.context_length)
#         mask.fill_(float("-inf"))
#         mask.triu_(1)  # 上三角为 -inf
#         return mask

#     def forward(self, text):
#         """
#         text: [Batch_size, context_length] (LongTensor)
#         """
#         # 1. Token Embedding + Positional Embedding
#         # x shape: [B, L, C]
#         x = self.token_embedding(text) + self.positional_embedding
        
#         # 2. Transformer 层 (CLIP 原始实现通常是 [L, B, C] 输入)
#         x = x.permute(1, 0, 2)  # [L, B, C]
#         x = self.transformer(x)
#         x = x.permute(1, 0, 2)  # [B, L, C]

#         # 3. Final LayerNorm
#         x = self.ln_final(x)

#         # 4. 提取 EOS Token 的特征
#         # 在 CLIP 中，text 通常以 EOS token 结尾，通过 argmax 找到最大 token id 的位置（即 EOS）
#         # x[batch, index] -> [B, C]
#         # 注意：text.argmax(dim=-1) 是一种常用 trick，假设 EOS token 的数值最大
#         # 或者在预处理时已知 EOS 位置
#         all_token_features = x @ self.text_projection

#         # 4. 从投影后的结果中提取 EOS 位置的特征
#         # text.argmax(dim=-1) 得到每个样本 EOS token 的索引
#         eos_indices = text.argmax(dim=-1)
#         pooled_output = all_token_features[torch.arange(all_token_features.shape[0]), eos_indices]

#         return all_token_features, pooled_output
class CLIPTextContextEncoder1(nn.Module):
    def __init__(self, context_length=24,
                 vocab_size=49408,
                 transformer_width=512,
                 transformer_heads=8,
                 transformer_layers=12,
                 embed_dim=1024,
                 out_dim=1024,
                 text_len=15,
                 # learnable_len=5,
                 learnable_len=5,
                 vitae=False,
                 pretrained=None, **kwargs):
        super().__init__()

        self.pretrained = pretrained
        self.vitae = vitae
        self.transformer_width = transformer_width
        self.context_length = context_length
        self.text_len = text_len
        self.learnable_len = learnable_len
        self.total_len = self.text_len + self.learnable_len
        self.positional_embedding = nn.Parameter(torch.empty(self.total_len, transformer_width))
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()  # 使用 total_len
        )

        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        
        # learnable tokens
        self.learnable_tokens = nn.Parameter(torch.randn(self.learnable_len, transformer_width) * 0.02)
        
        
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

        nn.init.trunc_normal_(self.positional_embedding, std=0.01)
        nn.init.trunc_normal_(self.text_projection, std=0.02)

    def build_attention_mask(self, seq_len=None):
        """
        创建 causal attention mask
        seq_len: 实际序列长度（text_len + learnable_len）
        """
        if seq_len is None:
            seq_len = self.total_len
        mask = torch.empty(seq_len, seq_len, device=self.positional_embedding.device)
        mask.fill_(float('-inf'))
        mask.triu_(1)  # 上三角填充-inf, 下三角为0
        return mask

    def forward(self, text):
        """
        text: [B, L], 长度不超过 self.text_len
        返回: [B, embed_dim]
        """
        device = text.device
        B = text.shape[0]
    
        # -------- 1. truncate text --------
        seq_len = min(text.shape[1], self.text_len)
        text = text[:, :seq_len]  # [B, L]
    
        # -------- 2. token embedding --------
        x_text = self.token_embedding(text)  # [B, L, C]
        L, C = x_text.shape[1], x_text.shape[2]
    
        # -------- 3. batch 内统一随机插入位置 --------
        insert_pos = torch.randint(low=1, high=L, size=(1,)).item()  # 插入位置
        learnable = self.learnable_tokens.unsqueeze(0).expand(B, -1, -1)  # [B, learnable_len, C]
        x = torch.cat([x_text[:, :insert_pos, :], learnable, x_text[:, insert_pos:, :]], dim=1)
        max_len = x.shape[1]
    
        # -------- 4. positional embedding --------
        if max_len > self.positional_embedding.shape[0]:
            # 如果长度超过 total_len, pad positional embedding
            pe = torch.cat([
                self.positional_embedding,
                torch.zeros(max_len - self.positional_embedding.shape[0], C, device=device)
            ], dim=0)
        else:
            pe = self.positional_embedding[:max_len, :]
        x = x + pe  # [B, L + learnable_len, C]
    
        # -------- 5. transformer --------
        x = x.permute(1, 0, 2)  # [L', B, C]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # [B, L', C]
        x = self.ln_final(x)
        x = x @ self.text_projection  # [B, L', embed_dim]
        tokenp = x
        # -------- 6. eos feature --------
        with torch.no_grad():
            eos_pos = text.argmax(dim=-1)  # [B], 原文本中最大 token id
        eos_pos = eos_pos.clamp(max=max_len - 1)
        x = x[torch.arange(B, device=device), eos_pos]  # [B, embed_dim]
        
        return tokenp,x



