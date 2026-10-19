# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmpose.registry import MODELS

from torch import Tensor

@MODELS.register_module()
class Loss2(nn.Module):
    """Combined Focal Loss and Online Hard Keypoint Mining for Heatmaps.
    
    Args:
        gamma (float): Focal focusing parameter. Defaults to 2.0.
        topk (int): Number of hardest keypoints to keep per sample. Defaults to 8.
        use_target_weight (bool): Whether to use keypoint-wise weights.
        loss_weight (float): Overall loss scaling factor.
    """

    def __init__(self,
                 gamma: float = 2.0,
                 topk: int = 8,
                 use_target_weight: bool = True,
                 loss_weight: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.topk = topk
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, 
                output: Tensor, 
                target: Tensor, 
                target_weights: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            output: [B, K, H, W]
            target: [B, K, H, W]
            target_weights: [B, K]
        """
        B, K, H, W = output.shape
        
        # 1. 计算元素级 MSE: [B, K, H, W]
        mse_per_pixel = F.mse_loss(output, target, reduction='none')

        # 2. 计算 Focal 权重: |target - output|^gamma
        # 这一步让每个关键点内部的“难点”像素贡献更多梯度
        with torch.no_grad():
            focal_weight = torch.abs(target - output).pow(self.gamma)
        
        # 3. 组合基础像素损失
        pixel_wise_loss = focal_weight * mse_per_pixel

        # 4. 如果使用 target_weights (关键点可见性/权重)
        if self.use_target_weight and target_weights is not None:
            # target_weights shape: [B, K] -> [B, K, 1, 1]
            pixel_wise_loss = pixel_wise_loss * target_weights.view(B, K, 1, 1)

        # 5. 计算每个关键点的平均损失: [B, K]
        # 将 H, W 维度的损失求平均，得到每个通道（关键点）的得分
        kp_loss = pixel_wise_loss.mean(dim=(2, 3))

        # 6. OHKM (Online Hard Keypoint Mining)
        # 在 K 个维度中挑选损失最大的 topk 个
        if self.topk < K:
            # topk_losses shape: [B, topk]
            topk_losses, _ = torch.topk(kp_loss, k=self.topk, dim=1, sorted=False)
            final_loss = topk_losses.mean() # 对 Batch 和 TopK 取平均
        else:
            final_loss = kp_loss.mean()

        return final_loss * self.loss_weight
@MODELS.register_module()
class KeypointMSELoss1(nn.Module):
    """改进的MSE损失，支持自适应权重和beta参数

    Args:
        use_target_weight (bool): 是否使用关键点权重
        skip_empty_channel (bool): 是否跳过空通道
        loss_weight (float): 损失权重
        beta (float): 控制损失函数的形状参数
                     beta=0: 标准MSE
                     beta>0: 强化大误差区域的惩罚
    """

    def __init__(self,
                 use_target_weight: bool = False,
                 skip_empty_channel: bool = False,
                 loss_weight: float = 1.,
                 beta: float = 0.0):
        super().__init__()
        self.use_target_weight = use_target_weight
        self.skip_empty_channel = skip_empty_channel
        self.loss_weight = loss_weight
        self.beta = beta

    def forward(self,
                output: Tensor,
                target: Tensor,
                target_weights: Optional[Tensor] = None,
                mask: Optional[Tensor] = None) -> Tensor:
        """前向传播函数
        
        改进点:
        1. 使用自适应权重: weight = |output - target|
        2. 引入beta参数: loss = weight^beta * (output - target)^2
        3. 当beta=0时退化为标准MSE
        
        Args:
            output (Tensor): 预测热图 [B, K, H, W]
            target (Tensor): 目标热图 [B, K, H, W]
            target_weights (Tensor, optional): 关键点权重 [B, K] 或 [B, K, H, W]
            mask (Tensor, optional): 像素级掩码 [B, K, H, W] 或 [B, 1, H, W]

        Returns:
            Tensor: 计算后的损失
        """
        # 获取组合掩码
        _mask = self._get_mask(target, target_weights, mask)
        
        # 计算残差
        diff = output - target
        
        # 核心改进：引入自适应权重机制
        if self.beta > 0:
            # 计算自适应权重并防止数值不稳定
            weight = torch.abs(diff) + 1e-8
            base_loss = weight.pow(self.beta) * (diff ** 2)
        else:
            # beta=0时退化为标准MSE
            base_loss = diff ** 2

        # 应用掩码
        if _mask is not None:
            loss = (base_loss * _mask).mean()
        else:
            loss = base_loss.mean()

        return loss * self.loss_weight

    def _get_mask(self, target: Tensor, target_weights: Optional[Tensor],
                  mask: Optional[Tensor]) -> Optional[Tensor]:
        """生成热图掩码（与原版保持一致）"""
        # 给定空间掩码
        if mask is not None:
            # 检查掩码与目标的匹配性
            assert (mask.ndim == target.ndim and all(
                d_m == d_t or d_m == 1
                for d_m, d_t in zip(mask.shape, target.shape))), (
                    f'mask和target形状不匹配 {mask.shape} v.s.'
                    f'{target.shape}')

        # 目标权重掩码（关键点级）
        if target_weights is not None:
            # 检查目标权重形状
            assert (target_weights.ndim in (2, 4) and target_weights.shape
                    == target.shape[:target_weights.ndim]), (
                        'target_weights和target形状不匹配 '
                        f'{target_weights.shape} v.s. {target.shape}')

            ndim_pad = target.ndim - target_weights.ndim
            _mask = target_weights.view(target_weights.shape +
                                        (1, ) * ndim_pad)

            if mask is None:
                mask = _mask
            else:
                mask = mask * _mask

        # 跳过空通道掩码
        if self.skip_empty_channel:
            _mask = (target != 0).flatten(2).any(dim=2)
            ndim_pad = target.ndim - _mask.ndim
            _mask = _mask.view(_mask.shape + (1, ) * ndim_pad)

            if mask is None:
                mask = _mask
            else:
                mask = mask * _mask

        return mask


@MODELS.register_module()
class KeypointMSELoss(nn.Module):
    """MSE loss for heatmaps.

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
            Defaults to ``False``
        skip_empty_channel (bool): If ``True``, heatmap channels with no
            non-zero value (which means no visible ground-truth keypoint
            in the image) will not be used to calculate the loss. Defaults to
            ``False``
        loss_weight (float): Weight of the loss. Defaults to 1.0
    """

    def __init__(self,
                 use_target_weight: bool = False,
                 skip_empty_channel: bool = False,
                 loss_weight: float = 1.):
        super().__init__()
        self.use_target_weight = use_target_weight
        self.skip_empty_channel = skip_empty_channel
        self.loss_weight = loss_weight

    def forward(self,
                output: Tensor,
                target: Tensor,
                target_weights: Optional[Tensor] = None,
                mask: Optional[Tensor] = None) -> Tensor:
        """Forward function of loss.

        Note:
            - batch_size: B
            - num_keypoints: K
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            output (Tensor): The output heatmaps with shape [B, K, H, W]
            target (Tensor): The target heatmaps with shape [B, K, H, W]
            target_weights (Tensor, optional): The target weights of differet
                keypoints, with shape [B, K] (keypoint-wise) or
                [B, K, H, W] (pixel-wise).
            mask (Tensor, optional): The masks of valid heatmap pixels in
                shape [B, K, H, W] or [B, 1, H, W]. If ``None``, no mask will
                be applied. Defaults to ``None``

        Returns:
            Tensor: The calculated loss.
        """

        _mask = self._get_mask(target, target_weights, mask)
        if _mask is None:
            loss = F.mse_loss(output, target)
        else:
            _loss = F.mse_loss(output, target, reduction='none')
            loss = (_loss * _mask).mean()

        return loss * self.loss_weight

    def _get_mask(self, target: Tensor, target_weights: Optional[Tensor],
                  mask: Optional[Tensor]) -> Optional[Tensor]:
        """Generate the heatmap mask w.r.t. the given mask, target weight and
        `skip_empty_channel` setting.

        Returns:
            Tensor: The mask in shape (B, K, *) or ``None`` if no mask is
            needed.
        """
        # Given spatial mask
        if mask is not None:
            # check mask has matching type with target
            assert (mask.ndim == target.ndim and all(
                d_m == d_t or d_m == 1
                for d_m, d_t in zip(mask.shape, target.shape))), (
                    f'mask and target have mismatched shapes {mask.shape} v.s.'
                    f'{target.shape}')

        # Mask by target weights (keypoint-wise mask)
        if target_weights is not None:
            # check target weight has matching shape with target
            assert (target_weights.ndim in (2, 4) and target_weights.shape
                    == target.shape[:target_weights.ndim]), (
                        'target_weights and target have mismatched shapes '
                        f'{target_weights.shape} v.s. {target.shape}')

            ndim_pad = target.ndim - target_weights.ndim
            _mask = target_weights.view(target_weights.shape +
                                        (1, ) * ndim_pad)

            if mask is None:
                mask = _mask
            else:
                mask = mask * _mask

        # Mask by ``skip_empty_channel``
        if self.skip_empty_channel:
            _mask = (target != 0).flatten(2).any(dim=2)
            ndim_pad = target.ndim - _mask.ndim
            _mask = _mask.view(_mask.shape + (1, ) * ndim_pad)

            if mask is None:
                mask = _mask
            else:
                mask = mask * _mask

        return mask


@MODELS.register_module()
class CombinedTargetMSELoss(nn.Module):
    """MSE loss for combined target.

    CombinedTarget: The combination of classification target
    (response map) and regression target (offset map).
    Paper ref: Huang et al. The Devil is in the Details: Delving into
    Unbiased Data Processing for Human Pose Estimation (CVPR 2020).

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
            Defaults to ``False``
        loss_weight (float): Weight of the loss. Defaults to 1.0
    """

    def __init__(self,
                 use_target_weight: bool = False,
                 loss_weight: float = 1.):
        super().__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, output: Tensor, target: Tensor,
                target_weights: Tensor) -> Tensor:
        """Forward function of loss.

        Note:
            - batch_size: B
            - num_channels: C
            - heatmaps height: H
            - heatmaps weight: W
            - num_keypoints: K
            Here, C = 3 * K

        Args:
            output (Tensor): The output feature maps with shape [B, C, H, W].
            target (Tensor): The target feature maps with shape [B, C, H, W].
            target_weights (Tensor): The target weights of differet keypoints,
                with shape [B, K].

        Returns:
            Tensor: The calculated loss.
        """
        batch_size = output.size(0)
        num_channels = output.size(1)
        heatmaps_pred = output.reshape(
            (batch_size, num_channels, -1)).split(1, 1)
        heatmaps_gt = target.reshape(
            (batch_size, num_channels, -1)).split(1, 1)
        loss = 0.
        num_joints = num_channels // 3
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx * 3].squeeze()
            heatmap_gt = heatmaps_gt[idx * 3].squeeze()
            offset_x_pred = heatmaps_pred[idx * 3 + 1].squeeze()
            offset_x_gt = heatmaps_gt[idx * 3 + 1].squeeze()
            offset_y_pred = heatmaps_pred[idx * 3 + 2].squeeze()
            offset_y_gt = heatmaps_gt[idx * 3 + 2].squeeze()
            if self.use_target_weight:
                target_weight = target_weights[:, idx, None]
                heatmap_pred = heatmap_pred * target_weight
                heatmap_gt = heatmap_gt * target_weight
            # classification loss
            loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)
            # regression loss
            loss += 0.5 * self.criterion(heatmap_gt * offset_x_pred,
                                         heatmap_gt * offset_x_gt)
            loss += 0.5 * self.criterion(heatmap_gt * offset_y_pred,
                                         heatmap_gt * offset_y_gt)
        return loss / num_joints * self.loss_weight


@MODELS.register_module()
class KeypointOHKMMSELoss(nn.Module):
    """MSE loss with online hard keypoint mining.

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
            Defaults to ``False``
        topk (int): Only top k joint losses are kept. Defaults to 8
        loss_weight (float): Weight of the loss. Defaults to 1.0
    """

    def __init__(self,
                 use_target_weight: bool = False,
                 topk: int = 8,
                 loss_weight: float = 1.):
        super().__init__()
        assert topk > 0
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.topk = topk
        self.loss_weight = loss_weight

    def _ohkm(self, losses: Tensor) -> Tensor:
        """Online hard keypoint mining.

        Note:
            - batch_size: B
            - num_keypoints: K

        Args:
            loss (Tensor): The losses with shape [B, K]

        Returns:
            Tensor: The calculated loss.
        """
        ohkm_loss = 0.
        B = losses.shape[0]
        for i in range(B):
            sub_loss = losses[i]
            _, topk_idx = torch.topk(
                sub_loss, k=self.topk, dim=0, sorted=False)
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / self.topk
        ohkm_loss /= B
        return ohkm_loss

    def forward(self, output: Tensor, target: Tensor,
                target_weights: Tensor) -> Tensor:
        """Forward function of loss.

        Note:
            - batch_size: B
            - num_keypoints: K
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            output (Tensor): The output heatmaps with shape [B, K, H, W].
            target (Tensor): The target heatmaps with shape [B, K, H, W].
            target_weights (Tensor): The target weights of differet keypoints,
                with shape [B, K].

        Returns:
            Tensor: The calculated loss.
        """
        num_keypoints = output.size(1)
        if num_keypoints < self.topk:
            raise ValueError(f'topk ({self.topk}) should not be '
                             f'larger than num_keypoints ({num_keypoints}).')

        losses = []
        for idx in range(num_keypoints):
            if self.use_target_weight:
                target_weight = target_weights[:, idx, None, None]
                losses.append(
                    self.criterion(output[:, idx] * target_weight,
                                   target[:, idx] * target_weight))
            else:
                losses.append(self.criterion(output[:, idx], target[:, idx]))

        losses = [loss.mean(dim=(1, 2)).unsqueeze(dim=1) for loss in losses]
        losses = torch.cat(losses, dim=1)

        return self._ohkm(losses) * self.loss_weight


@MODELS.register_module()
class AdaptiveWingLoss(nn.Module):
    """Adaptive wing loss. paper ref: 'Adaptive Wing Loss for Robust Face
    Alignment via Heatmap Regression' Wang et al. ICCV'2019.

    Args:
        alpha (float), omega (float), epsilon (float), theta (float)
            are hyper-parameters.
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self,
                 alpha=2.1,
                 omega=14,
                 epsilon=1,
                 theta=0.5,
                 use_target_weight=False,
                 loss_weight=1.):
        super().__init__()
        self.alpha = float(alpha)
        self.omega = float(omega)
        self.epsilon = float(epsilon)
        self.theta = float(theta)
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def criterion(self, pred, target):
        """Criterion of wingloss.

        Note:
            batch_size: N
            num_keypoints: K

        Args:
            pred (torch.Tensor[NxKxHxW]): Predicted heatmaps.
            target (torch.Tensor[NxKxHxW]): Target heatmaps.
        """
        H, W = pred.shape[2:4]
        delta = (target - pred).abs()

        A = self.omega * (
            1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - target))
        ) * (self.alpha - target) * (torch.pow(
            self.theta / self.epsilon,
            self.alpha - target - 1)) * (1 / self.epsilon)
        C = self.theta * A - self.omega * torch.log(
            1 + torch.pow(self.theta / self.epsilon, self.alpha - target))

        losses = torch.where(
            delta < self.theta,
            self.omega *
            torch.log(1 +
                      torch.pow(delta / self.epsilon, self.alpha - target)),
            A * delta - C)

        return torch.mean(losses)

    def forward(self,
                output: Tensor,
                target: Tensor,
                target_weights: Optional[Tensor] = None):
        """Forward function.

        Note:
            batch_size: N
            num_keypoints: K

        Args:
            output (torch.Tensor[N, K, H, W]): Output heatmaps.
            target (torch.Tensor[N, K, H, W]): Target heatmaps.
            target_weight (torch.Tensor[N, K]):
                Weights across different joint types.
        """
        if self.use_target_weight:
            assert (target_weights.ndim in (2, 4) and target_weights.shape
                    == target.shape[:target_weights.ndim]), (
                        'target_weights and target have mismatched shapes '
                        f'{target_weights.shape} v.s. {target.shape}')

            ndim_pad = target.ndim - target_weights.ndim
            target_weights = target_weights.view(target_weights.shape +
                                                 (1, ) * ndim_pad)
            loss = self.criterion(output * target_weights,
                                  target * target_weights)
        else:
            loss = self.criterion(output, target)

        return loss * self.loss_weight


@MODELS.register_module()
class FocalHeatmapLoss(KeypointMSELoss):
    """A class for calculating the modified focal loss for heatmap prediction.

    This loss function is exactly the same as the one used in CornerNet. It
    runs faster and costs a little bit more memory.

    `CornerNet: Detecting Objects as Paired Keypoints
    arXiv: <https://arxiv.org/abs/1808.01244>`_.

    Arguments:
        alpha (int): The alpha parameter in the focal loss equation.
        beta (int): The beta parameter in the focal loss equation.
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
            Defaults to ``False``
        skip_empty_channel (bool): If ``True``, heatmap channels with no
            non-zero value (which means no visible ground-truth keypoint
            in the image) will not be used to calculate the loss. Defaults to
            ``False``
        loss_weight (float): Weight of the loss. Defaults to 1.0
    """

    def __init__(self,
                 alpha: int = 2,
                 beta: int = 4,
                 use_target_weight: bool = False,
                 skip_empty_channel: bool = False,
                 loss_weight: float = 1.0):
        super(FocalHeatmapLoss, self).__init__(use_target_weight,
                                               skip_empty_channel, loss_weight)
        self.alpha = alpha
        self.beta = beta

    def forward(self,
                output: Tensor,
                target: Tensor,
                target_weights: Optional[Tensor] = None,
                mask: Optional[Tensor] = None) -> Tensor:
        """Calculate the modified focal loss for heatmap prediction.

        Note:
            - batch_size: B
            - num_keypoints: K
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            output (Tensor): The output heatmaps with shape [B, K, H, W]
            target (Tensor): The target heatmaps with shape [B, K, H, W]
            target_weights (Tensor, optional): The target weights of differet
                keypoints, with shape [B, K] (keypoint-wise) or
                [B, K, H, W] (pixel-wise).
            mask (Tensor, optional): The masks of valid heatmap pixels in
                shape [B, K, H, W] or [B, 1, H, W]. If ``None``, no mask will
                be applied. Defaults to ``None``

        Returns:
            Tensor: The calculated loss.
        """
        _mask = self._get_mask(target, target_weights, mask)

        pos_inds = target.eq(1).float()
        neg_inds = target.lt(1).float()

        if _mask is not None:
            pos_inds = pos_inds * _mask
            neg_inds = neg_inds * _mask

        neg_weights = torch.pow(1 - target, self.beta)

        pos_loss = torch.log(output) * torch.pow(1 - output,
                                                 self.alpha) * pos_inds
        neg_loss = torch.log(1 - output) * torch.pow(
            output, self.alpha) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        if num_pos == 0:
            loss = -neg_loss.sum()
        else:
            loss = -(pos_loss.sum() + neg_loss.sum()) / num_pos
        return loss * self.loss_weight


@MODELS.register_module()
class MLECCLoss(nn.Module):
    """Maximum Likelihood Estimation loss for Coordinate Classification.

    This loss function is designed to work with coordinate classification
    problems where the likelihood of each target coordinate is maximized.

    Args:
        reduction (str): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. Default: 'mean'.
        mode (str): Specifies the mode of calculating loss:
            'linear' | 'square' | 'log'. Default: 'log'.
        use_target_weight (bool): If True, uses weighted loss. Different
            joint types may have different target weights. Defaults to False.
        loss_weight (float): Weight of the loss. Defaults to 1.0.

    Raises:
        AssertionError: If the `reduction` or `mode` arguments are not in the
                        expected choices.
        NotImplementedError: If the selected mode is not implemented.
    """

    def __init__(self,
                 reduction: str = 'mean',
                 mode: str = 'log',
                 use_target_weight: bool = False,
                 loss_weight: float = 1.0):
        super().__init__()
        assert reduction in ('mean', 'sum', 'none'), \
            f"`reduction` should be either 'mean', 'sum', or 'none', " \
            f'but got {reduction}'
        assert mode in ('linear', 'square', 'log'), \
            f"`mode` should be either 'linear', 'square', or 'log', " \
            f'but got {mode}'

        self.reduction = reduction
        self.mode = mode
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, outputs, targets, target_weight=None):
        """Forward pass for the MLECCLoss.

        Args:
            outputs (torch.Tensor): The predicted outputs.
            targets (torch.Tensor): The ground truth targets.
            target_weight (torch.Tensor, optional): Optional tensor of weights
                for each target.

        Returns:
            torch.Tensor: Calculated loss based on the specified mode and
                reduction.
        """

        assert len(outputs) == len(targets), \
            'Outputs and targets must have the same length'

        prob = 1.0
        for o, t in zip(outputs, targets):
            prob *= (o * t).sum(dim=-1)

        if self.mode == 'linear':
            loss = 1.0 - prob
        elif self.mode == 'square':
            loss = 1.0 - prob.pow(2)
        elif self.mode == 'log':
            loss = -torch.log(prob + 1e-4)

        loss[torch.isnan(loss)] = 0.0

        if self.use_target_weight:
            assert target_weight is not None
            for i in range(loss.ndim - target_weight.ndim):
                target_weight = target_weight.unsqueeze(-1)
            loss = loss * target_weight

        if self.reduction == 'sum':
            loss = loss.flatten(1).sum(dim=1)
        elif self.reduction == 'mean':
            loss = loss.flatten(1).mean(dim=1)

        return loss * self.loss_weight
