# Copyright (c) OpenMMLab. All rights reserved.
import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine import MessageHub
from mmengine.dist import barrier, broadcast, get_dist_info
from mmengine.structures import PixelData
from torch import Tensor

from mmpose.registry import MODELS
from mmpose.structures import PoseDataSample
import numpy as np

@MODELS.register_module()
class BatchSyncRandomResize(nn.Module):
    """Batch random resize which synchronizes the random size across ranks.

    Args:
        random_size_range (tuple): The multi-scale random range during
            multi-scale training.
        interval (int): The iter interval of change
            image size. Defaults to 10.
        size_divisor (int): Image size divisible factor.
            Defaults to 32.
    """

    def __init__(self,
                 random_size_range: Tuple[int, int],
                 interval: int = 10,
                 size_divisor: int = 32) -> None:
        super().__init__()
        self.rank, self.world_size = get_dist_info()
        self._input_size = None
        self._random_size_range = (round(random_size_range[0] / size_divisor),
                                   round(random_size_range[1] / size_divisor))
        self._interval = interval
        self._size_divisor = size_divisor

    def forward(self, inputs: Tensor, data_samples: List[PoseDataSample]
                ) -> Tuple[Tensor, List[PoseDataSample]]:
        """resize a batch of images and bboxes to shape ``self._input_size``"""
        h, w = inputs.shape[-2:]
        if self._input_size is None:
            self._input_size = (h, w)
        scale_y = self._input_size[0] / h
        scale_x = self._input_size[1] / w
        if scale_x != 1 or scale_y != 1:
            inputs = F.interpolate(
                inputs,
                size=self._input_size,
                mode='bilinear',
                align_corners=False)
            for data_sample in data_samples:
                img_shape = (int(data_sample.img_shape[0] * scale_y),
                             int(data_sample.img_shape[1] * scale_x))
                pad_shape = (int(data_sample.pad_shape[0] * scale_y),
                             int(data_sample.pad_shape[1] * scale_x))
                data_sample.set_metainfo({
                    'img_shape': img_shape,
                    'pad_shape': pad_shape,
                    'batch_input_shape': self._input_size
                })

                if 'gt_instance_labels' not in data_sample:
                    continue

                if 'bboxes' in data_sample.gt_instance_labels:
                    data_sample.gt_instance_labels.bboxes[..., 0::2] *= scale_x
                    data_sample.gt_instance_labels.bboxes[..., 1::2] *= scale_y

                if 'keypoints' in data_sample.gt_instance_labels:
                    data_sample.gt_instance_labels.keypoints[..., 0] *= scale_x
                    data_sample.gt_instance_labels.keypoints[..., 1] *= scale_y

                if 'areas' in data_sample.gt_instance_labels:
                    data_sample.gt_instance_labels.areas *= scale_x * scale_y

                if 'gt_fields' in data_sample \
                        and 'heatmap_mask' in data_sample.gt_fields:

                    mask = data_sample.gt_fields.heatmap_mask.unsqueeze(0)
                    gt_fields = PixelData()
                    gt_fields.set_field(
                        F.interpolate(
                            mask.float(),
                            size=self._input_size,
                            mode='bilinear',
                            align_corners=False).squeeze(0), 'heatmap_mask')

                    data_sample.gt_fields = gt_fields

        message_hub = MessageHub.get_current_instance()
        if (message_hub.get_info('iter') + 1) % self._interval == 0:
            self._input_size = self._get_random_size(
                aspect_ratio=float(w / h), device=inputs.device)
        return inputs, data_samples

    def _get_random_size(self, aspect_ratio: float,
                         device: torch.device) -> Tuple[int, int]:
        """Randomly generate a shape in ``_random_size_range`` and broadcast to
        all ranks."""
        tensor = torch.LongTensor(2).to(device)
        if self.rank == 0:
            size = random.randint(*self._random_size_range)
            size = (self._size_divisor * size,
                    self._size_divisor * int(aspect_ratio * size))
            tensor[0] = size[0]
            tensor[1] = size[1]
        barrier()
        broadcast(tensor, 0)
        input_size = (tensor[0].item(), tensor[1].item())
        return input_size
@MODELS.register_module()
class KeypointCutMix(nn.Module):
    def __init__(self, mask_joint_num=4):
        super().__init__()
        self.mask_joint_num = mask_joint_num

    def forward(self, inputs: Tensor, data_samples: List[PoseDataSample]) -> Tuple[Tensor, List[PoseDataSample]]:
        image = inputs.clone()
        N, C, H, W = image.shape

        joints = []
        # for ds in data_samples:
        #     print(ds.gt_instance_labels)

        for ds in data_samples:
            gt_labels = ds.gt_instance_labels  # type: InstanceData

            #print(ds.gt_instances.keys())
  # 查看所有字段名

            keypoints = ds.gt_instances.get('keypoints', None)
            if keypoints is None:
                print("***")
                continue
            joints.append(keypoints)
        joints = [torch.tensor(j) if isinstance(j, np.ndarray) else j for j in joints]
        joints = torch.stack(joints).to(image.device)
        joints = joints.squeeze(1)
        # print(f'joints.shape = {joints.shape}')

        # shape: (N, J, 3)

        J = joints.shape[1]
        # re_joints = joints[:, :, :2] + torch.randn((N, J, 2), device=image.device) * 10
        # re_joints = re_joints.int()
        re_joints = joints[:, :, :2] + torch.randn_like(joints[:, :, :2]) * 10
        re_joints = re_joints.int()

        size = torch.randint(10, 20, (N, J, 2), device=image.device).int()

        cx = re_joints[:, :, 0]
        cy = re_joints[:, :, 1]
        x0 = torch.clamp(cx - size[:, :, 0], 0, W - 1)
        y0 = torch.clamp(cy - size[:, :, 1], 0, H - 1)
        x1 = torch.clamp(cx + size[:, :, 0], 0, W - 1)
        y1 = torch.clamp(cy + size[:, :, 1], 0, H - 1)

        for i in range(N):
            ind = torch.randperm(J)[:self.mask_joint_num]
            ind_2 = torch.randperm(J)[:self.mask_joint_num]
            img_id = random.randint(0, N - 1)
        
            for idx in range(self.mask_joint_num):
                j = ind[idx].item()  # 强制转int
                j2 = ind_2[idx].item()
        
                # # 调试打印类型和形状
                # print(f'i type: {type(i)}, value: {i}')
                # print(f'j type: {type(j)}, value: {j}, shape: {j.shape if isinstance(j, torch.Tensor) else "N/A"}')
                # print(f'j2 type: {type(j2)}, value: {j2}, shape: {j2.shape if isinstance(j2, torch.Tensor) else "N/A"}')
                # print(f'img_id type: {type(img_id)}, value: {img_id}')
        
                # # 如果 j 和 j2 是 tensor，转成 int
                # if isinstance(j, torch.Tensor):
                #     if j.numel() == 1:
                #         j = j.item()
                #     else:
                #         print(f"ERROR: j tensor with more than 1 element: {j}")
                #         raise RuntimeError("j tensor has more than 1 element")
                # if isinstance(j2, torch.Tensor):
                #     if j2.numel() == 1:
                #         j2 = j2.item()
                #     else:
                #         print(f"ERROR: j2 tensor with more than 1 element: {j2}")
                #         raise RuntimeError("j2 tensor has more than 1 element")
        
                # # 同理 img_id
                # if isinstance(img_id, torch.Tensor):
                #     if img_id.numel() == 1:
                #         img_id = img_id.item()
                #     else:
                #         print(f"ERROR: img_id tensor with more than 1 element: {img_id}")
                #         raise RuntimeError("img_id tensor has more than 1 element")
        
                cx_i_j = re_joints[i, j, 0].item()
                cy_i_j = re_joints[i, j, 1].item()
                # print(f'cx_i_j shape: {cx_i_j.shape if isinstance(cx_i_j, torch.Tensor) else "N/A"}, value: {cx_i_j}')
                
                # # 如果这里还是Tensor且元素数>1，说明索引i,j不对
                # if isinstance(cx_i_j, torch.Tensor) and cx_i_j.numel() != 1:
                #     print(f"ERROR: cx_i_j has more than 1 element: {cx_i_j}")
                #     raise RuntimeError("cx_i_j tensor has more than 1 element")
                # if isinstance(cx_i_j, torch.Tensor):
                #     cx_i_j = cx_i_j.item()
                # if isinstance(cy_i_j, torch.Tensor):
                #     cy_i_j = cy_i_j.item()
        
                # 这里类似地处理其它变量...
                cx_img_j2 = re_joints[img_id, j2, 0].item()
                cy_img_j2 = re_joints[img_id, j2, 1].item()
        
                x0_img_j2 = x0[img_id, j2].item()
                y0_img_j2 = y0[img_id, j2].item()
                x1_img_j2 = x1[img_id, j2].item()
                y1_img_j2 = y1[img_id, j2].item()
        
                # 剩余代码保持不变
                xs = cx_i_j - abs(cx_img_j2 - x0_img_j2)
                xe = cx_i_j + abs(x1_img_j2 - cx_img_j2)
                ys = cy_i_j - abs(cy_img_j2 - y0_img_j2)
                ye = cy_i_j + abs(y1_img_j2 - cy_img_j2)
        
                xs = int(max(0, min(xs, W - 1)))
                xe = int(max(0, min(xe, W - 1)))
                ys = int(max(0, min(ys, H - 1)))
                ye = int(max(0, min(ye, H - 1)))
        
                patch_src = image[img_id, :, y0_img_j2:y1_img_j2, x0_img_j2:x1_img_j2]
                patch_tgt = image[i, :, ys:ye, xs:xe]
        
                if patch_src.shape[-2:] != patch_tgt.shape[-2:] or patch_src.numel() == 0 or patch_tgt.numel() == 0:
                    continue
                image[i, :, ys:ye, xs:xe] = patch_src
        return image, data_samples






