# Copyright (c) OpenMMLab. All rights reserved.
import copy
from abc import ABCMeta
from collections import defaultdict
from typing import Optional, Sequence, Tuple
import random
import numpy as np
import torch
from typing import Optional, Sequence
from torchvision.transforms import Compose
import mmcv
import numpy as np
from mmcv.transforms import BaseTransform
from mmengine.dataset.base_dataset import Compose
from numpy import random

from mmpose.registry import TRANSFORMS
from mmpose.structures import (bbox_clip_border, flip_bbox, flip_keypoints,
                               keypoint_clip_border)


class MixImageTransform(BaseTransform, metaclass=ABCMeta):
    """Abstract base class for mixup-style image data augmentation.

    Args:
        pre_transform (Optional[Sequence[str]]): A sequence of transform
            to be applied before mixup. Defaults to None.
        prob (float): Probability of applying the mixup transformation.
            Defaults to 1.0.
    """

    def __init__(self,
                 pre_transform: Optional[Sequence[str]] = None,
                 prob: float = 1.0):

        self.prob = prob

        if pre_transform is None:
            self.pre_transform = None
        else:
            self.pre_transform = Compose(pre_transform)

    def transform(self, results: dict) -> dict:
        """Transform the input data dictionary using mixup-style augmentation.

        Args:
            results (dict): A dictionary containing input data.
        """

        if random.uniform(0, 1) < self.prob:

            dataset = results.pop('dataset', None)

            results['mixed_data_list'] = self._get_mixed_data_list(dataset)
            results = self.apply_mix(results)

            if 'mixed_data_list' in results:
                results.pop('mixed_data_list')

            results['dataset'] = dataset

        return results

    def _get_mixed_data_list(self, dataset):
        """Get a list of mixed data samples from the dataset.

        Args:
            dataset: The dataset from which to sample the mixed data.

        Returns:
            List[dict]: A list of dictionaries containing mixed data samples.
        """
        indexes = [
            random.randint(0, len(dataset)) for _ in range(self.num_aux_image)
        ]

        mixed_data_list = [
            copy.deepcopy(dataset.get_data_info(index)) for index in indexes
        ]

        if self.pre_transform is not None:
            for i, data in enumerate(mixed_data_list):
                data.update({'dataset': dataset})
                _results = self.pre_transform(data)
                _results.pop('dataset')
                mixed_data_list[i] = _results

        return mixed_data_list
# @TRANSFORMS.register_module()
# class CutMixTransform(MixImageTransform):
#     """CutMix数据增强：混合两张图像的部分区域
    
#     Args:
#         pre_transform (Optional[Sequence[str]]): 前置变换序列
#         prob (float): 应用概率，默认1.0（总是应用）
#         alpha (float): Beta分布参数，控制混合强度
#         num_aux_image (int): 辅助图像数量，默认1
#         use_keypoint_bbox (bool): 是否基于关键点生成切割区域
    
#     MMPose数据增强规范：
#     - 继承MixImageTransform基类
#     - 使用TRANSFORMS注册机制
#     - 明确声明修改和新增的keys
#     - 支持配置文件调用
#     """

#     def __init__(self,
#                  pre_transform: Optional[Sequence[str]] = None,
#                  prob: float = 1.0,
#                  alpha: float = 1.0,
#                  num_aux_image: int = 1,
#                  use_keypoint_bbox: bool = False):
#         super().__init__(pre_transform, prob)
#         self.alpha = alpha
#         self.num_aux_image = num_aux_image
#         self.use_keypoint_bbox = use_keypoint_bbox

#     def apply_mix(self, results: dict) -> dict:
#         """应用CutMix混合操作
        
#         Required Keys:
#             - img
#             - img_shape
#             - keypoints (可选)
#             - keypoints_visible (可选)
        
#         Modified Keys:
#             - img
#             - keypoints_visible (当use_keypoint_bbox=True时)
        
#         Added Keys:
#             - lam (混合比例)
#         """
#         # 获取主图像数据
#         img_main = results['img']
#         h, w = img_main.shape[1:3]
        
#         # 获取辅助图像数据
#         data_aux = results['mixed_data_list'][0]
#         img_aux = data_aux['img']
        
#         # 生成lambda参数（混合比例）
#         lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1.0
        
#         # 生成切割区域
#         if self.use_keypoint_bbox and 'keypoints' in results:
#             bbx1, bby1, bbx2, bby2 = self._keypoint_based_bbox(
#                 results['keypoints'], 
#                 results['keypoints_visible'],
#                 h, w
#             )
#         else:
#             bbx1, bby1, bbx2, bby2 = self._rand_bbox(h, w, lam)
        
#         # 执行图像混合
#         img_main[:, bby1:bby2, bbx1:bbx2] = img_aux[:, bby1:bby2, bbx1:bbx2]
        
#         # 更新关键点可见性（如果使用关键点引导）
#         if self.use_keypoint_bbox and 'keypoints' in results:
#             results = self._update_keypoint_visibility(
#                 results, 
#                 bbx1, bby1, bbx2, bby2
#             )
        
#         # 存储混合比例
#         results['lam'] = lam
#         return results

#     def _rand_bbox(self, h: int, w: int, lam: float) -> Tuple[int, int, int, int]:
#         """生成随机切割区域"""
#         cut_rat = np.sqrt(1 - lam)
#         cut_w = int(w * cut_rat)
#         cut_h = int(h * cut_rat)
        
#         cx = np.random.randint(w)
#         cy = np.random.randint(h)
        
#         bbx1 = np.clip(cx - cut_w // 2, 0, w)
#         bby1 = np.clip(cy - cut_h // 2, 0, h)
#         bbx2 = np.clip(cx + cut_w // 2, 0, w)
#         bby2 = np.clip(cy + cut_h // 2, 0, h)
        
#         return bbx1, bby1, bbx2, bby2

#     def _keypoint_based_bbox(self, keypoints: np.ndarray, 
#                             visibility: np.ndarray, 
#                             h: int, w: int) -> Tuple[int, int, int, int]:
#         """基于关键点位置生成切割区域"""
#         # 过滤可见关键点
#         visible_kpts = keypoints[visibility > 0]
        
#         if len(visible_kpts) > 0:
#             # 随机选择一个可见关键点作为中心
#             center_idx = np.random.randint(0, len(visible_kpts))
#             cx, cy = visible_kpts[center_idx]
#         else:
#             # 无可见关键点时使用图像中心
#             cx, cy = w // 2, h // 2
        
#         # 随机确定切割区域大小
#         area_ratio = np.random.uniform(0.05, 0.25)
#         cut_size = int(np.sqrt(area_ratio * h * w))
        
#         # 计算边界（确保在图像范围内）
#         bbx1 = max(0, int(cx - cut_size // 2))
#         bby1 = max(0, int(cy - cut_size // 2))
#         bbx2 = min(w, int(cx + cut_size // 2))
#         bby2 = min(h, int(cy + cut_size // 2))
        
#         return bbx1, bby1, bbx2, bby2

#     def _update_keypoint_visibility(self, results: dict,
#                                   bbx1: int, bby1: int, 
#                                   bbx2: int, bby2: int) -> dict:
#         """更新被遮挡关键点的可见性"""
#         keypoints = results['keypoints']
#         visibility = results['keypoints_visible']
        
#         # 检查关键点是否位于切割区域内
#         for i in range(len(keypoints)):
#             kx, ky = keypoints[i]
#             if (bbx1 <= kx <= bbx2) and (bby1 <= ky <= bby2):
#                 visibility[i] = 0  # 被遮挡的关键点设为不可见
        
#         # 更新可见性权重
#         if 'keypoint_weights' in results:
#             results['keypoint_weights'] = results['keypoint_weights'] * visibility
        
#         return results

#     def __repr__(self) -> str:
#         repr_str = super().__repr__()
#         repr_str += f'(alpha={self.alpha}, num_aux_image={self.num_aux_image}'
#         repr_str += f', use_keypoint_bbox={self.use_keypoint_bbox})'
#         return repr_str
# @TRANSFORMS.register_module()

# class Cutout(MixImageTransform):
#     """Cutout数据增强：随机遮挡图像中的矩形区域
    
#     Args:
#         pre_transform (Optional[Sequence[str]]): 前置变换序列
#         prob (float): 应用概率，默认1.0（总是应用）
#         n_holes (int): 遮挡区域数量
#         length (int): 正方形遮挡区域边长
#         fill_value (int): 填充值（0或数据集均值）
#     """
#     def __init__(self, 
#                  pre_transform: Optional[Sequence[str]] = None,
#                  prob: float = 1.0,
#                  n_holes: int = 1,
#                  length: int = 16,
#                  fill_value: int = 0):
#         super().__init__(pre_transform, prob)
#         self.n_holes = n_holes
#         self.length = length
#         self.fill_value = fill_value
#         self.num_aux_image = 0  # Cutout无需辅助图像

#     def apply_mix(self, results: dict) -> dict:
#         """应用Cutout到主图像"""
#         img = results['img']
        
#         # 获取图像高度和宽度（兼容不同格式）
#         if len(img.shape) == 3:  # CHW格式（张量）或 HWC格式（numpy）
#             h, w = img.shape[1], img.shape[2] if isinstance(img, torch.Tensor) else img.shape[0], img.shape[1]
#         else:  # 二维灰度图
#             h, w = img.shape

#         # 创建全1遮挡掩码 (H, W)
#         mask = np.ones((h, w), np.float32)
        
#         # 生成随机遮挡区域
#         for _ in range(self.n_holes):
#             y = np.random.randint(h)  # 随机中心点y坐标
#             x = np.random.randint(w)  # 随机中心点x坐标
            
#             # 计算矩形边界（确保不越界）
#             y1 = np.clip(y - self.length // 2, 0, h)
#             y2 = np.clip(y + self.length // 2, 0, h)
#             x1 = np.clip(x - self.length // 2, 0, w)
#             x2 = np.clip(x + self.length // 2, 0, w)
            
#             mask[y1:y2, x1:x2] = 0.  # 遮挡区域置0
        
#         # 确保图像是张量后再进行expand_as操作--------------------关键修改点--------------------
#         if not isinstance(img, torch.Tensor):
#             img_tensor = torch.from_numpy(img).float()
#         else:
#             img_tensor = img
        
#         # 扩展掩码维度至(C, H, W) 
#         mask_tensor = torch.from_numpy(mask).float()
        
#         # 扩展掩码以匹配图像形状
#         if len(img_tensor.shape) == 3:
#             # 添加通道维度
#             mask_tensor = mask_tensor.unsqueeze(0)  # 变为 1xHxW
#             # 扩展通道数
#             mask_tensor = mask_tensor.expand_as(img_tensor)  # 变为 CxHxW
#         else:
#             mask_tensor = mask_tensor.expand_as(img_tensor)  # 保持 HxW
        
#         # 应用遮挡：原始图像 * 掩码 + 填充值 * 反掩码
#         results['img'] = img_tensor * mask_tensor + self.fill_value * (1 - mask_tensor)
        
#         return results

#     def __repr__(self) -> str:
#         return (f"{self.__class__.__name__}(n_holes={self.n_holes}, "
#                 f"length={self.length}, fill_value={self.fill_value})")
@TRANSFORMS.register_module()
class Mosaic(MixImageTransform):
    """Mosaic augmentation. This transformation takes four input images and
    combines them into a single output image using the mosaic technique. The
    resulting image is composed of parts from each of the four sub-images. The
    mosaic transform steps are as follows:

    1. Choose the mosaic center as the intersection of the four images.
    2. Select the top-left image according to the index and randomly sample
        three more images from the custom dataset.
    3. If an image is larger than the mosaic patch, it will be cropped.

    .. code:: text

                        mosaic transform
                           center_x
                +------------------------------+
                |       pad        |           |
                |      +-----------+    pad    |
                |      |           |           |
                |      |  image1   +-----------+
                |      |           |           |
                |      |           |   image2  |
     center_y   |----+-+-----------+-----------+
                |    |   cropped   |           |
                |pad |   image3    |   image4  |
                |    |             |           |
                +----|-------------+-----------+
                     |             |
                     +-------------+

    Required Keys:

    - img
    - bbox (optional)
    - bbox_score (optional)
    - category_id (optional)
    - keypoints (optional)
    - keypoints_visible (optional)
    - area (optional)

    Modified Keys:

    - img
    - bbox (optional)
    - bbox_score (optional)
    - category_id (optional)
    - keypoints (optional)
    - keypoints_visible (optional)
    - area (optional)

    Args:
        img_scale (Sequence[int]): Image size after mosaic pipeline of single
            image. The shape order should be (width, height).
            Defaults to (640, 640).
        center_range (Sequence[float]): Center ratio range of mosaic
            output. Defaults to (0.5, 1.5).
        pad_val (int): Pad value. Defaults to 114.
        pre_transform (Optional[Sequence[str]]): A sequence of transform
            to be applied before mixup. Defaults to None.
        prob (float): Probability of applying the mixup transformation.
            Defaults to 1.0.
    """

    num_aux_image = 3

    def __init__(
        self,
        img_scale: Tuple[int, int] = (640, 640),
        center_range: Tuple[float, float] = (0.5, 1.5),
        pad_val: float = 114.0,
        pre_transform: Sequence[dict] = None,
        prob: float = 1.0,
    ):

        super().__init__(pre_transform=pre_transform, prob=prob)

        self.img_scale = img_scale
        self.center_range = center_range
        self.pad_val = pad_val

    def apply_mix(self, results: dict) -> dict:
        """Apply mosaic augmentation to the input data."""

        assert 'mixed_data_list' in results
        mixed_data_list = results.pop('mixed_data_list')
        assert len(mixed_data_list) == self.num_aux_image

        img, annos = self._create_mosaic_image(results, mixed_data_list)
        bboxes = annos['bboxes']
        kpts = annos['keypoints']
        kpts_vis = annos['keypoints_visible']

        bboxes = bbox_clip_border(bboxes, (2 * self.img_scale[0],
                                           2 * self.img_scale[1]))
        kpts, kpts_vis = keypoint_clip_border(kpts, kpts_vis,
                                              (2 * self.img_scale[0],
                                               2 * self.img_scale[1]))

        results['img'] = img
        results['img_shape'] = img.shape
        results['bbox'] = bboxes
        results['category_id'] = annos['category_id']
        results['bbox_score'] = annos['bbox_scores']
        results['keypoints'] = kpts
        results['keypoints_visible'] = kpts_vis
        results['area'] = annos['area']

        return results

    def _create_mosaic_image(self, results, mixed_data_list):
        """Create the mosaic image and corresponding annotations by combining
        four input images."""

        # init mosaic image
        img_scale_w, img_scale_h = self.img_scale
        mosaic_img = np.full((int(img_scale_h * 2), int(img_scale_w * 2), 3),
                             self.pad_val,
                             dtype=results['img'].dtype)

        # calculate mosaic center
        center = (int(random.uniform(*self.center_range) * img_scale_w),
                  int(random.uniform(*self.center_range) * img_scale_h))

        annos = defaultdict(list)
        locs = ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        for loc, data in zip(locs, (results, *mixed_data_list)):

            # process image
            img = data['img']
            h, w = img.shape[:2]
            scale_ratio = min(img_scale_h / h, img_scale_w / w)
            img = mmcv.imresize(img,
                                (int(w * scale_ratio), int(h * scale_ratio)))

            # paste
            paste_coord, crop_coord = self._mosaic_combine(
                loc, center, img.shape[:2][::-1])
            x1_p, y1_p, x2_p, y2_p = paste_coord
            x1_c, y1_c, x2_c, y2_c = crop_coord

            # crop and paste image
            mosaic_img[y1_p:y2_p, x1_p:x2_p] = img[y1_c:y2_c, x1_c:x2_c]
            padw = x1_p - x1_c
            padh = y1_p - y1_c

            # merge annotations
            if 'bbox' in data:
                bboxes = data['bbox']

                # rescale & translate
                bboxes *= scale_ratio
                bboxes[..., ::2] += padw
                bboxes[..., 1::2] += padh

                annos['bboxes'].append(bboxes)
                annos['bbox_scores'].append(data['bbox_score'])
                annos['category_id'].append(data['category_id'])

            if 'keypoints' in data:
                kpts = data['keypoints']

                # rescale & translate
                kpts *= scale_ratio
                kpts[..., 0] += padw
                kpts[..., 1] += padh

                annos['keypoints'].append(kpts)
                annos['keypoints_visible'].append(data['keypoints_visible'])

            if 'area' in data:
                annos['area'].append(data['area'] * scale_ratio**2)

        for key in annos:
            annos[key] = np.concatenate(annos[key])
        return mosaic_img, annos

    def _mosaic_combine(
        self, loc: str, center: Tuple[float, float], img_shape: Tuple[int, int]
    ) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
        """Determine the overall coordinates of the mosaic image and the
        specific coordinates of the cropped sub-image."""

        assert loc in ('top_left', 'top_right', 'bottom_left', 'bottom_right')

        x1, y1, x2, y2 = 0, 0, 0, 0
        cx, cy = center
        w, h = img_shape

        if loc == 'top_left':
            x1, y1, x2, y2 = max(cx - w, 0), max(cy - h, 0), cx, cy
            crop_coord = w - (x2 - x1), h - (y2 - y1), w, h
        elif loc == 'top_right':
            x1, y1, x2, y2 = cx, max(cy - h, 0), min(cx + w,
                                                     self.img_scale[0] * 2), cy
            crop_coord = 0, h - (y2 - y1), min(w, x2 - x1), h
        elif loc == 'bottom_left':
            x1, y1, x2, y2 = max(cx - w,
                                 0), cy, cx, min(self.img_scale[1] * 2, cy + h)
            crop_coord = w - (x2 - x1), 0, w, min(y2 - y1, h)
        else:
            x1, y1, x2, y2 = cx, cy, min(cx + w, self.img_scale[0] *
                                         2), min(self.img_scale[1] * 2, cy + h)
            crop_coord = 0, 0, min(w, x2 - x1), min(y2 - y1, h)

        return (x1, y1, x2, y2), crop_coord

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'center_range={self.center_range}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'prob={self.prob})'
        return repr_str


@TRANSFORMS.register_module()
class YOLOXMixUp(MixImageTransform):
    """MixUp data augmentation for YOLOX. This transform combines two images
    through mixup to enhance the dataset's diversity.

    Mixup Transform Steps:

        1. A random image is chosen from the dataset and placed in the
            top-left corner of the target image (after padding and resizing).
        2. The target of the mixup transform is obtained by taking the
            weighted average of the mixup image and the original image.

    .. code:: text

                         mixup transform
                +---------------+--------------+
                | mixup image   |              |
                |      +--------|--------+     |
                |      |        |        |     |
                +---------------+        |     |
                |      |                 |     |
                |      |      image      |     |
                |      |                 |     |
                |      |                 |     |
                |      +-----------------+     |
                |             pad              |
                +------------------------------+

    Required Keys:

    - img
    - bbox (optional)
    - bbox_score (optional)
    - category_id (optional)
    - keypoints (optional)
    - keypoints_visible (optional)
    - area (optional)

    Modified Keys:

    - img
    - bbox (optional)
    - bbox_score (optional)
    - category_id (optional)
    - keypoints (optional)
    - keypoints_visible (optional)
    - area (optional)

    Args:
        img_scale (Sequence[int]): Image output size after mixup pipeline.
            The shape order should be (width, height). Defaults to (640, 640).
        ratio_range (Sequence[float]): Scale ratio of mixup image.
            Defaults to (0.5, 1.5).
        flip_ratio (float): Horizontal flip ratio of mixup image.
            Defaults to 0.5.
        pad_val (int): Pad value. Defaults to 114.
        pre_transform (Optional[Sequence[str]]): A sequence of transform
            to be applied before mixup. Defaults to None.
        prob (float): Probability of applying the mixup transformation.
            Defaults to 1.0.
    """
    num_aux_image = 1

    def __init__(self,
                 img_scale: Tuple[int, int] = (640, 640),
                 ratio_range: Tuple[float, float] = (0.5, 1.5),
                 flip_ratio: float = 0.5,
                 pad_val: float = 114.0,
                 bbox_clip_border: bool = True,
                 pre_transform: Sequence[dict] = None,
                 prob: float = 1.0):
        assert isinstance(img_scale, tuple)
        super().__init__(pre_transform=pre_transform, prob=prob)
        self.img_scale = img_scale
        self.ratio_range = ratio_range
        self.flip_ratio = flip_ratio
        self.pad_val = pad_val
        self.bbox_clip_border = bbox_clip_border

    def apply_mix(self, results: dict) -> dict:
        """YOLOX MixUp transform function."""

        assert 'mixed_data_list' in results
        mixed_data_list = results.pop('mixed_data_list')
        assert len(mixed_data_list) == self.num_aux_image

        if mixed_data_list[0]['keypoints'].shape[0] == 0:
            return results

        img, annos = self._create_mixup_image(results, mixed_data_list)
        bboxes = annos['bboxes']
        kpts = annos['keypoints']
        kpts_vis = annos['keypoints_visible']

        h, w = img.shape[:2]
        bboxes = bbox_clip_border(bboxes, (w, h))
        kpts, kpts_vis = keypoint_clip_border(kpts, kpts_vis, (w, h))

        results['img'] = img.astype(np.uint8)
        results['img_shape'] = img.shape
        results['bbox'] = bboxes
        results['category_id'] = annos['category_id']
        results['bbox_score'] = annos['bbox_scores']
        results['keypoints'] = kpts
        results['keypoints_visible'] = kpts_vis
        results['area'] = annos['area']

        return results

    def _create_mixup_image(self, results, mixed_data_list):
        """Create the mixup image and corresponding annotations by combining
        two input images."""

        aux_results = mixed_data_list[0]
        aux_img = aux_results['img']

        # init mixup image
        out_img = np.ones((self.img_scale[1], self.img_scale[0], 3),
                          dtype=aux_img.dtype) * self.pad_val
        annos = defaultdict(list)

        # Calculate scale ratio and resize aux_img
        scale_ratio = min(self.img_scale[1] / aux_img.shape[0],
                          self.img_scale[0] / aux_img.shape[1])
        aux_img = mmcv.imresize(aux_img, (int(aux_img.shape[1] * scale_ratio),
                                          int(aux_img.shape[0] * scale_ratio)))

        # Set the resized aux_img in the top-left of out_img
        out_img[:aux_img.shape[0], :aux_img.shape[1]] = aux_img

        # random rescale
        jit_factor = random.uniform(*self.ratio_range)
        scale_ratio *= jit_factor
        out_img = mmcv.imresize(out_img, (int(out_img.shape[1] * jit_factor),
                                          int(out_img.shape[0] * jit_factor)))

        # random flip
        is_filp = random.uniform(0, 1) > self.flip_ratio
        if is_filp:
            out_img = out_img[:, ::-1, :]

        # random crop
        ori_img = results['img']
        aux_h, aux_w = out_img.shape[:2]
        h, w = ori_img.shape[:2]
        padded_img = np.ones((max(aux_h, h), max(aux_w, w), 3)) * self.pad_val
        padded_img = padded_img.astype(np.uint8)
        padded_img[:aux_h, :aux_w] = out_img

        dy = random.randint(0, max(0, padded_img.shape[0] - h) + 1)
        dx = random.randint(0, max(0, padded_img.shape[1] - w) + 1)
        padded_cropped_img = padded_img[dy:dy + h, dx:dx + w]

        # mix up
        mixup_img = 0.5 * ori_img + 0.5 * padded_cropped_img

        # merge annotations
        # bboxes
        bboxes = aux_results['bbox'].copy()
        bboxes *= scale_ratio
        bboxes = bbox_clip_border(bboxes, (aux_w, aux_h))
        if is_filp:
            bboxes = flip_bbox(bboxes, [aux_w, aux_h], 'xyxy')
        bboxes[..., ::2] -= dx
        bboxes[..., 1::2] -= dy
        annos['bboxes'] = [results['bbox'], bboxes]
        annos['bbox_scores'] = [
            results['bbox_score'], aux_results['bbox_score']
        ]
        annos['category_id'] = [
            results['category_id'], aux_results['category_id']
        ]

        # keypoints
        kpts = aux_results['keypoints'] * scale_ratio
        kpts, kpts_vis = keypoint_clip_border(kpts,
                                              aux_results['keypoints_visible'],
                                              (aux_w, aux_h))
        if is_filp:
            kpts, kpts_vis = flip_keypoints(kpts, kpts_vis, (aux_w, aux_h),
                                            aux_results['flip_indices'])
        kpts[..., 0] -= dx
        kpts[..., 1] -= dy
        annos['keypoints'] = [results['keypoints'], kpts]
        annos['keypoints_visible'] = [results['keypoints_visible'], kpts_vis]
        annos['area'] = [results['area'], aux_results['area'] * scale_ratio**2]

        for key in annos:
            annos[key] = np.concatenate(annos[key])

        return mixup_img, annos

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'flip_ratio={self.flip_ratio}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str
