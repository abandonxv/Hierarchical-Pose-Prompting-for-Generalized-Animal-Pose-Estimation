import torch
import torch.nn as nn
from mmcv.cnn.bricks import DropPath
from torch import Tensor

class ConvModule(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, conv_cfg=None, norm_cfg=None, act_cfg=None):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)]
        if norm_cfg and norm_cfg.get('type') == 'BN':
            layers.append(nn.BatchNorm2d(out_channels, momentum=norm_cfg.get('momentum', 0.1)))
        if act_cfg:
            layers.append(nn.ReLU())
        super().__init__(*layers)

class BaseModule(nn.Module):

    def __init__(self, init_cfg=None):
        super().__init__()

class MaxSigmoidAttnBlock(BaseModule):

    def __init__(self, in_channels, out_channels, guide_channels, embed_channels, num_heads=1, with_scale=False, use_einsum=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_channels = embed_channels // num_heads
        self.use_einsum = use_einsum
        self.embed_conv = nn.Conv2d(in_channels, embed_channels, 1) if embed_channels != in_channels else None
        self.guide_fc = nn.Linear(guide_channels, embed_channels)
        self.bias = nn.Parameter(torch.zeros(num_heads))
        self.scale = nn.Parameter(torch.ones(1, num_heads, 1, 1)) if with_scale else 1.0
        self.project_conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.LayerNorm(embed_channels)
        self.norm2 = nn.LayerNorm(embed_channels)
        self.returntext = True

    def forward(self, x: Tensor, guide: Tensor) -> Tensor:
        B, C, H, W = x.shape
        B, N, D = guide.shape
        guide = self.norm1(guide)
        guide = self.guide_fc(guide)
        guide1 = guide
        guide = guide.reshape(B, -1, self.num_heads, self.head_channels)
        x_ = x.permute(0, 2, 3, 1)
        x_ = self.norm2(x_)
        x = x_.permute(0, 3, 1, 2)
        embed = self.embed_conv(x) if self.embed_conv is not None else x
        embed = embed.reshape(B, self.num_heads, self.head_channels, H, W)
        if self.use_einsum:
            attn_weight = torch.einsum('bmchw,bnmc->bmhwn', embed, guide)
            attnmap = attn_weight
        else:
            batch, m, channel, height, width = embed.shape
            _, n, _, _ = guide.shape
            embed = embed.permute(0, 1, 3, 4, 2)
            guide = guide.permute(0, 2, 3, 1)
            attn_weight = torch.matmul(embed, guide)
        attn_weight = attn_weight.max(dim=-1)[0]
        attn_weight = attn_weight / self.head_channels ** 0.5
        attn_weight = attn_weight + self.bias[None, :, None, None]
        attn_weight = attn_weight.sigmoid() * self.scale
        x = self.project_conv(x)
        x = x.reshape(B, self.num_heads, -1, H, W)
        x = x * attn_weight.unsqueeze(2)
        x = x.reshape(B, -1, H, W)
        if self.returntext:
            q = guide1
            k = x.reshape(B, -1, C)
            v = k
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / C ** 0.5
            attn_probs = torch.softmax(attn_scores, dim=-1)
            text_updated = torch.matmul(attn_probs, v)
            return (x, text_updated)
        else:
            return (x, attnmap)

class MaxSigmoidAttnBlockWithRes(nn.Module):
    """
    Wrapper: 在不修改原始 MaxSigmoidAttnBlock 的前提下
    添加 residual + DropPath + LayerScale
    """

    def __init__(self, in_channels, out_channels, guide_channels, embed_channels, num_heads=8, drop_path=0.0, layer_scale_init=1e-05):
        super().__init__()
        self.block = MaxSigmoidAttnBlock(in_channels, out_channels, guide_channels, embed_channels, num_heads=num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.gamma_img = nn.Parameter(layer_scale_init * torch.ones(out_channels))
        self.gamma_txt = nn.Parameter(layer_scale_init * torch.ones(embed_channels))

    def forward(self, x, text):
        identity_x = x
        identity_t = text
        res = self.block(x, text)
        if len(res) == 3:
            x_out, text_out, attn_map = res
        elif len(res) == 2:
            x_out, attn_map = res
            text_out = None
        x = identity_x + self.drop_path(self.gamma_img.view(1, -1, 1, 1) * x_out)
        if text_out is not None:
            text = identity_t + self.drop_path(self.gamma_txt.view(1, 1, -1) * text_out)
        return (x, text)

class MaxSigmoidAttnBlockStack(nn.Module):

    def __init__(self, in_channels, out_channels, guide_channels, embed_channels, depth=3, num_heads=8, drop_path=0.2, layer_scale_init=1e-05):
        super().__init__()
        dpr = torch.linspace(0, drop_path, depth)
        self.layers = nn.ModuleList([MaxSigmoidAttnBlockWithRes(in_channels=in_channels, out_channels=out_channels, guide_channels=guide_channels, embed_channels=embed_channels, num_heads=num_heads, drop_path=dpr[i].item(), layer_scale_init=layer_scale_init) for i in range(depth)])

    def forward(self, x, text):
        all_attn_maps = []
        for layer in self.layers:
            x, text = layer(x, text)
        return (x, text)
