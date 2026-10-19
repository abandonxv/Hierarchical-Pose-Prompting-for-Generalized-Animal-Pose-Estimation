from itertools import zip_longest
from typing import List, Optional

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from torch import Tensor

from mmpose.models import builder
from mmpose.registry import MODELS
from mmpose.utils.typing import (ConfigType, InstanceList, OptConfigType,
                                 OptMultiConfig, PixelDataList, SampleList)

from ..backbones.textenc1 import PromptSimplifier
from .base import BasePoseEstimator
from .kpa3 import ConcatAttentionMatcherStack
from .pra3 import MaxSigmoidAttnBlockStack
from .untils import tokenize

class TripleDeconvUpsample(nn.Module):

    def __init__(self, in_channels, num_filters):
        """
        num_filters: list，如 [256, 128]，表示每次上采样后的目标通道数
        """
        super().__init__()
        layers = []
        for out_channels in num_filters:
            layers.append(nn.Conv2d(in_channels, out_channels, 1))
            layers.append(nn.GroupNorm(1, out_channels))
            layers.append(nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False))
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(QuickGELU())
            in_channels = out_channels
        self.upsample = nn.Sequential(*layers)

    def forward(self, x):
        x = self.upsample(x)
        return x

class LayerNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-06, data_format='channels_last'):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ['channels_last', 'channels_first']:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == 'channels_last':
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == 'channels_first':
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class QuickGELU(nn.Module):

    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)

class LayerScale(nn.Module):
    """LayerScale: learnable scaling factor for residual"""

    def __init__(self, dim, init_value=1e-05):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma

class LayerScaledim(nn.Module):

    def __init__(self, dim, init_value=1e-05):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x):
        return self.gamma[None, :, None, None] * x

@MODELS.register_module()
class hpp3(BasePoseEstimator):

    def __init__(self, backbone: ConfigType, text_encoder: ConfigType, class_names: ConfigType, neck: OptConfigType=None, head: OptConfigType=None, head1: OptConfigType=None, train_cfg: OptConfigType=None, test_cfg: OptConfigType=None, data_preprocessor: OptConfigType=None, init_cfg: OptMultiConfig=None, metainfo: Optional[dict]=None, context_length=13, token_embed_dim=512, clip_pretrained=None, pretrained=None):
        super().__init__(backbone=backbone, neck=neck, head=head, train_cfg=train_cfg, test_cfg=test_cfg, data_preprocessor=data_preprocessor, init_cfg=init_cfg, metainfo=metainfo)
        self.backbone = builder.build_backbone(backbone)
        self.text_encoder = builder.build_backbone(text_encoder)
        self.head1 = builder.build_head(head1)

        @property
        def with_head1(self):
            return hasattr(self, 'head1') and self.head1 is not None
        self.class_names = class_names
        self.context_length = context_length
        self.num_classes = len(class_names)
        self.keypoint = 17
        self.texts = torch.cat([tokenize(c, context_length=self.context_length) for c in class_names])
        self.contexts = nn.Parameter(torch.randn(1, 5, token_embed_dim))
        nn.init.trunc_normal_(self.contexts)
        self.invisible_kpt_token = nn.Parameter(torch.empty(1, 1, token_embed_dim))
        nn.init.trunc_normal_(self.invisible_kpt_token, mean=0.0, std=0.02, a=-0.04, b=0.04)
        self.uphead = TripleDeconvUpsample(in_channels=529, num_filters=[256, 128])
        self.uphead2 = TripleDeconvUpsample(in_channels=512, num_filters=[256, 128])
        self.uphead1 = TripleDeconvUpsample(in_channels=17, num_filters=[17, 17])
        self.conv1 = nn.Conv2d(512, 512, 1)
        self.conv2 = nn.Sequential(nn.Conv2d(128, 32, 1), nn.BatchNorm2d(32), nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32, bias=False), QuickGELU())
        self.layerscale3 = LayerScaledim(dim=17, init_value=1e-06)
        self.ls1 = LayerScaledim(dim=512, init_value=1e-06)
        self.ls2 = LayerScale(dim=512, init_value=1e-06)
        self.drop_path = DropPath(drop_prob=0.1)
        self.drop_path1 = DropPath(drop_prob=0.1)
        self.gloss = gloss()
        self.kloss = kloss()
        if self.text_encoder is not None:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
            self.text_encoder.eval()
        self.fuse = MaxSigmoidAttnBlockStack(in_channels=512, out_channels=512, guide_channels=512, embed_channels=512, num_heads=8)
        self.kpa = ConcatAttentionMatcherStack(embed_dims=512, depth=3, drop_path=0.1)
        self.proj = nn.Linear(512 // 2, 512)
        self.simple = PromptSimplifier()

    def safe_tokenize(self, texts: List[str]):
        out = clip.tokenize(texts, context_length=24)
        if not isinstance(out, torch.Tensor):
            out = torch.tensor(out, dtype=torch.long)
        return out

    def extract_feat(self, inputs: Tensor, data_samples: SampleList) -> SampleList:
        text_feat = None
        if data_samples is not None:
            text_prompts = [s.metainfo.get('instance_text_prompt', '') for s in data_samples]
            if any((p.strip() == '' for p in text_prompts)):
                raise ValueError('Empty text prompt detected')
            text_prompts = [self.simple.simplify(t) for t in text_prompts]
            text_tokens = self.safe_tokenize(text_prompts).to(inputs.device)
            text_feat, _ = self.text_encoder(text_tokens)
        x = self.backbone(inputs)
        cls_token, visual_embeddings = x[0]
        b, c, h, w = visual_embeddings.shape
        guidetext = []
        par, guidetext = self.fuse(visual_embeddings, text_feat)
        visual_embeddings = visual_embeddings + self.drop_path(self.ls1(par))
        visual_embeddings = visual_embeddings.reshape(b, -1, c)
        vis = visual_embeddings
        _, kprompt = self.text_encoder(self.texts.to(cls_token.device))
        keypoints_visible = torch.stack([torch.as_tensor(ds.gt_instances.keypoints_visible, device=inputs.device, dtype=torch.float32) for ds in data_samples], dim=0)
        keypoints_visible = keypoints_visible.squeeze(1)
        keypoints_visible = keypoints_visible.to(device=kprompt.device, dtype=kprompt.dtype)
        visible = keypoints_visible.unsqueeze(-1).type_as(kprompt)
        kprompt = visible * kprompt + (1.0 - visible) * self.invisible_kpt_token
        kpa = self.kpa(visual_embeddings, kprompt)
        visual_embeddings = kpa
        visual_embeddings = visual_embeddings.reshape(b, -1, h, w)
        visual_embeddings_norm = F.normalize(visual_embeddings, dim=1, p=2)
        prompt_embeddings_norm = F.normalize(kprompt, dim=2, p=2)
        score_map = torch.einsum('bchw,bkc->bkhw', visual_embeddings_norm, prompt_embeddings_norm)
        kembed = score_map
        score_map = self.uphead1(score_map)
        output = self.conv2(self.uphead2(visual_embeddings))
        if torch.jit.is_tracing():
            return (output, score_map)
        else:
            return (output, score_map, guidetext, vis, kprompt, kembed)

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        keypoints_visible = torch.stack([torch.as_tensor(ds.gt_instances.keypoints_visible, device=inputs.device, dtype=torch.float32) for ds in data_samples], dim=0)
        keypoints_visible = keypoints_visible.squeeze(1)
        kptvis = keypoints_visible
        output, score_map, guidetext, vis, kprompt, kembed = self.extract_feat(inputs, data_samples)
        losses = dict()
        gloss = 1
        closs = 0
        if gloss:
            losses['gloss'] = self.gloss(vis, guidetext) * 0.0002
        if closs:
            losses['kloss'] = self.kloss(kembed, kprompt) * 5e-05
        losses1 = self.head1.loss(score_map, data_samples, train_cfg=self.train_cfg)
        losses.update({f'head1_{k}': v for k, v in losses1.items()})
        if self.with_head:
            losses.update(self.head.loss(output, data_samples, train_cfg=self.train_cfg))
        return losses

    def predict(self, inputs: Tensor, data_samples: SampleList):
        assert self.with_head, 'The model must have head to perform prediction.'
        if self.test_cfg.get('flip_test', False):
            _feats, _, _, _, _, _ = self.extract_feat(inputs, data_samples)
            _feats_flip, _, _, _, _, _ = self.extract_feat(inputs.flip(-1), data_samples)
            feats = [_feats, _feats_flip]
        else:
            feats, _, _, _, _, _ = self.extract_feat(inputs, data_samples)
        preds = self.head.predict(feats, data_samples, test_cfg=self.test_cfg)
        if isinstance(preds, tuple):
            batch_pred_instances, batch_pred_fields = preds
        else:
            batch_pred_instances = preds
            batch_pred_fields = None
        results = self.add_pred_to_datasample(batch_pred_instances, batch_pred_fields, data_samples)
        return results

    def add_pred_to_datasample(self, batch_pred_instances: InstanceList, batch_pred_fields: Optional[PixelDataList], batch_data_samples: SampleList) -> SampleList:
        assert len(batch_pred_instances) == len(batch_data_samples)
        if batch_pred_fields is None:
            batch_pred_fields = []
        output_keypoint_indices = self.test_cfg.get('output_keypoint_indices', None)
        for pred_instances, pred_fields, data_sample in zip_longest(batch_pred_instances, batch_pred_fields, batch_data_samples):
            gt_instances = data_sample.gt_instances
            input_center = data_sample.metainfo['input_center']
            input_scale = data_sample.metainfo['input_scale']
            input_size = data_sample.metainfo['input_size']
            pred_instances.keypoints[..., :2] = pred_instances.keypoints[..., :2] / input_size * input_scale + input_center - 0.5 * input_scale
            if 'keypoints_visible' not in pred_instances:
                pred_instances.keypoints_visible = pred_instances.keypoint_scores
            if output_keypoint_indices is not None:
                num_keypoints = pred_instances.keypoints.shape[1]
                for key, value in pred_instances.all_items():
                    if key.startswith('keypoint'):
                        pred_instances.set_field(value[:, output_keypoint_indices], key)
            pred_instances.bboxes = gt_instances.bboxes
            pred_instances.bbox_scores = gt_instances.bbox_scores
            data_sample.pred_instances = pred_instances
            if pred_fields is not None:
                if output_keypoint_indices is not None:
                    for key, value in pred_fields.all_items():
                        if value.shape[0] != num_keypoints:
                            continue
                        pred_fields.set_field(value[output_keypoint_indices], key)
                data_sample.pred_fields = pred_fields
        return batch_data_samples

class gloss(nn.Module):

    def __init__(self, visual_dim=512, text_dim=512, embed_dim=512, init_temp=0.07, loss_weight=0.05, entropy_weight=0.01):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / init_temp)))
        self.loss_weight = loss_weight
        self.entropy_weight = entropy_weight

    def forward(self, visual_embeddings, guidetext):
        """
        visual_embeddings : [B, N, C]   (pose tokens / image tokens)
        guidetext         : [B, T, D]   (text tokens)
        """
        B, N, C = visual_embeddings.shape
        text_feat = guidetext.mean(1)
        attn = torch.einsum('bd,bnd->bn', text_feat, visual_embeddings)
        attn = attn.softmax(dim=-1)
        pose_feat = torch.sum(visual_embeddings * attn.unsqueeze(-1), dim=1)
        pose_feat = F.normalize(pose_feat, dim=-1)
        text_feat = F.normalize(text_feat, dim=-1)
        pose_feat = pose_feat.float()
        text_feat = text_feat.float()
        logit_scale = self.logit_scale.exp().clamp(max=50)
        logits_per_image = logit_scale * pose_feat @ text_feat.t()
        logits_per_text = logits_per_image.t()
        labels = torch.arange(B, device=pose_feat.device)
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        contrast_loss = (loss_i + loss_t) / 2
        entropy = -(attn * torch.log(attn + 1e-06)).sum(-1).mean()
        entropy_loss = -entropy
        loss = self.loss_weight * contrast_loss + self.entropy_weight * entropy_loss
        return loss

class kloss(nn.Module):

    def __init__(self, temperature=0.07, global_weight=0.1, eps=1e-06):
        super().__init__()
        self.temperature = temperature
        self.global_weight = global_weight
        self.eps = eps
        self.proj = nn.Linear(256, 512)

    def forward(self, visual_feat, text_feat):
        visual_feat = visual_feat.flatten(2)
        visual_feat = self.proj(visual_feat)
        B, K, D = visual_feat.shape
        v = F.normalize(visual_feat, dim=-1, eps=self.eps)
        t = F.normalize(text_feat, dim=-1, eps=self.eps)
        sim = torch.bmm(v, t.transpose(1, 2)) / self.temperature
        labels = torch.arange(K, device=sim.device)
        labels = labels.unsqueeze(0).repeat(B, 1)
        loss_i2t = F.cross_entropy(sim, labels)
        loss_t2i = F.cross_entropy(sim.transpose(1, 2), labels)
        contrast_loss = 0.5 * (loss_i2t + loss_t2i)
        v_g = F.normalize(v.mean(1), dim=-1)
        t_g = F.normalize(t.mean(1), dim=-1)
        global_loss = 1 - (v_g * t_g).sum(-1).mean()
        loss = contrast_loss + self.global_weight * global_loss
        return loss
