import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

def build_rope_freq(seq_len, dim, device):
    """
    生成 RoPE 的频率向量。
    Args:
        seq_len: 序列长度 Nq 或 Nk
        dim: 每个 head 的维度 D
        device: torch.device
    Returns:
        freq: [seq_len, dim//2]
    """
    dim_half = dim // 2
    inv_freq = 1.0 / 10000 ** (torch.arange(0, dim_half, dtype=torch.float32, device=device) / dim_half)
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.einsum('i,j->ij', pos, inv_freq)
    return freqs

def apply_rope(x, rope_freq):
    """
    将 RoPE 作用于 q 或 k。
    Args:
        x: [B, H, N, D] 其中 D = dim_per_head
        rope_freq: [N, D//2]
    Returns:
        x_rot: [B, H, N, D]
    """
    B, H, N, D = x.shape
    assert D % 2 == 0, 'RoPE 需要偶数维度'
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    cos = torch.cos(rope_freq).unsqueeze(0).unsqueeze(0)
    sin = torch.sin(rope_freq).unsqueeze(0).unsqueeze(0)
    x_rot1 = x1 * cos - x2 * sin
    x_rot2 = x1 * sin + x2 * cos
    x_rot = torch.stack([x_rot1, x_rot2], dim=-1).reshape(B, H, N, D)
    return x_rot

class LayerScale(nn.Module):

    def __init__(self, dim, init_value=1e-05):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x):
        return self.gamma * x

class QuickGELU(nn.Module):

    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)

class DWConv(nn.Module):

    def __init__(self, dim=512):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x, H, W):
        x = self.dwconv(x)
        return x

class ConvolutionalGLU(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features)
        self.act = QuickGELU()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()
        x = self.fc1(x)
        x = self.act(x + self.dwconv(x))
        x = self.fc2(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.drop(x)
        return x

class RMSNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-06):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class DropKey(nn.Module):

    def __init__(self, drop_prob=0.1):
        super(DropKey, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, attn_scores):
        """
        Args:
            attn_scores: [Batch, Heads, Query_Len, Key_Len]
                         即 Softmax 之前的原始得分
        """
        if not self.training or self.drop_prob == 0.0:
            return attn_scores
        m_shape = (attn_scores.size(0), 1, 1, attn_scores.size(-1))
        keep_prob = 1 - self.drop_prob
        mask = torch.bernoulli(torch.full(m_shape, keep_prob, device=attn_scores.device))
        return attn_scores.masked_fill(mask == 0, float('-inf'))

class AttentionMatcher(nn.Module):

    def __init__(self, embed_dims, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0.1, proj_drop=0.1, norm_cfg=dict(type='LN')):
        super().__init__()
        self.num_heads = num_heads
        head_dim = embed_dims // num_heads
        self.scale = qk_scale or head_dim ** (-0.5)
        self.q = nn.Linear(embed_dims, embed_dims, bias=qkv_bias)
        self.k = nn.Linear(embed_dims, embed_dims, bias=qkv_bias)
        self.v = nn.Linear(embed_dims, embed_dims, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm = nn.LayerNorm(embed_dims)
        self.temperature = nn.Parameter(torch.log((torch.ones(self.num_heads, 1, 1) / 0.24).exp() - 1))
        self.query_embedding = nn.Parameter(nn.init.trunc_normal_(torch.empty(self.num_heads, 1, head_dim), mean=0, std=0.02))
        self.register_buffer('seq_length_scale', torch.as_tensor(np.log(16 * 16)), persistent=False)
        self.q_norm = RMSNorm(head_dim, eps=1e-06)
        self.k_norm = RMSNorm(head_dim, eps=1e-06)
        self.dkey = DropKey(drop_prob=0.05)

    def forward_attention(self, xq, xk, xv, return_attn=True):
        B, Nq, C = xq.size()
        Nk = xk.size(1)
        Nv = xv.size(1)
        D = C // self.num_heads
        q = self.q(xq).reshape(B, Nq, self.num_heads, D).permute(0, 2, 1, 3)
        k = self.k(xk).reshape(B, Nk, self.num_heads, D).permute(0, 2, 1, 3)
        v = self.v(xv).reshape(B, Nv, self.num_heads, D).permute(0, 2, 1, 3)
        qk_dtype = q.dtype
        q = self.q_norm(q).to(qk_dtype)
        k = self.k_norm(k).to(qk_dtype)
        rope_q = build_rope_freq(Nq, D, q.device)
        rope_k = build_rope_freq(Nk, D, k.device)
        q = apply_rope(q, rope_q)
        k = apply_rope(k, rope_k)
        q = F.normalize(q, dim=-1) + self.query_embedding
        k = F.normalize(k, dim=-1)
        scale = F.softplus(self.temperature).view(1, self.num_heads, 1, 1)
        scale = scale * self.seq_length_scale
        q = q * scale
        attn = q @ k.transpose(-2, -1)
        attn = self.dkey(attn)
        attn = attn.softmax(dim=-1)
        attn_vis = attn
        attn = self.attn_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, Nq, C)
        x = self.proj_drop(self.proj(x))
        if return_attn:
            return (x, attn_vis)
        else:
            return x

class ConcatAttentionMatcher(AttentionMatcher):

    def __init__(self, embed_dims, drop_path=0.1, layer_scale_init=1e-05, **kwargs):
        super().__init__(embed_dims, **kwargs)
        self.ffn = ConvolutionalGLU(in_features=embed_dims, hidden_features=embed_dims * 2, drop=0.1)
        self.drop_path1 = DropPath(drop_path)
        self.drop_path2 = DropPath(drop_path)
        self.ls1 = LayerScale(embed_dims, init_value=layer_scale_init)
        self.ls2 = LayerScale(embed_dims, init_value=layer_scale_init)
        self.norm2 = nn.LayerNorm(embed_dims)

    def forward(self, image_embed, text_embed, return_attn=False):
        out = self.forward_attention(image_embed, text_embed, text_embed, return_attn=False)
        x = image_embed + self.drop_path1(self.ls1(out))
        ffn_out = self.ffn(self.norm2(x), H=16, W=16)
        x = x + self.drop_path2(self.ls2(ffn_out))
        if return_attn:
            return (x, attn)
        else:
            return x

class ConcatAttentionMatcherStack(nn.Module):
    """
    Stacked ConcatAttentionMatcher
    """

    def __init__(self, embed_dims, depth=3, drop_path=0.2, layer_scale_init=1e-05):
        super().__init__()
        dpr = torch.linspace(0, drop_path, depth)
        self.layers = nn.ModuleList([ConcatAttentionMatcher(embed_dims=embed_dims, drop_path=dpr[i], layer_scale_init=layer_scale_init) for i in range(depth)])

    def forward(self, image_embed, text_embed):
        attn_maps = []
        for layer in self.layers:
            img = layer(image_embed, text_embed)
        return img
