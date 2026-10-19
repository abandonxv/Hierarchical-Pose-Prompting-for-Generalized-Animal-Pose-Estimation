_base_ = ['../../../_base_/default_runtime.py']

train_cfg = dict(max_epochs=210, val_interval=10, dynamic_intervals=[(190, 1)])
base_lr = 3e-4
# base_lr = 5e-4
max_epochs= 210
randomness = dict(seed=4)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=base_lr,  # 从 5e-4 降低至 2e-4 更稳定
        weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            'text_encoder': dict(lr_mult=0.0),  # 冻结文本编码器
            'backbone': dict(lr_mult=0.05),  # HRNet 等 backbone 学习率减小
            'norm': dict(decay_mult=0.0)  # norm 层不进行权重衰减
        }
    ),
    clip_grad=dict(
        max_norm=0.8,
        norm_type=2
    ))


# learning policy

param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001,
        by_epoch=False),  # warm-up
    # dict(
    #     type='MultiStepLR',
    #     begin=0,
    #     end=210,
    #     milestones=[170, 200],
    #     gamma=0.1,
    #     by_epoch=True)
    dict(
        type='CosineAnnealingLR',
        # eta_min=base_lr * 0.0001,
        eta_min=base_lr * 0.001,
        # begin=180,# 提前启动
        begin=max_epochs//2,
        # begin=max_epochs//3,
        end=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True)
]
auto_scale_lr = dict(base_batch_size=512)

# default_hooks = dict(checkpoint=dict(save_best='coco/AP', rule='greater'))
default_hooks = dict(
    checkpoint=dict(save_best='coco/AP', rule='greater', max_keep_ckpts=1))
codec = dict(
    type='MSRAHeatmap', input_size=(256, 256), heatmap_size=(64, 64), sigma=2)

# model settings
model = dict(
    type='hpp3',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    clip_pretrained='/root/mmpose/pretrained/ViT-B-16 .pt',
    context_length=15,
    # text_dim=512,
    # visual_dim=512,
    class_names=['left eye', 'right eye', 'nose', 'neck', 'tail root',
                 'left shoulder', 'left elbow', 'left front paw',
                 'right shoulder', 'right elbow', 'right front paw',
                 'left hip', 'left knee', 'left back paw',
                 'right hip', 'right knee', 'right back paw'],
    text_encoder=dict(
        type='CLIPTextContextEncoder1',
        context_length=15,
        embed_dim=512,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        pretrained='/root/mmpose/pretrained/ViT-B-16 .pt',
        style='pytorch'),
    backbone=dict(
        type='CLIPVisionTransformer',
        debug=False,
        use_fpn=False,
        patch_size=16,
        width=768,
        output_dim=512,
        get_embeddings=True,
        drop_path_rate=0.4,
        layers=12,
        pretrained='/root/mmpose/pretrained/ViT-B-16 .pt',
        input_resolution=256,
        style='pytorch'),
    # neck=dict(type='Identity'),
    # head=dict(type='Identity'),
    head=dict(
        type='HeatmapHead',
        in_channels=32,
        out_channels=17,
        deconv_out_channels=None,
        loss=dict(type='KeypointMSELoss1', use_target_weight=True),
        decoder=codec),
    head1=dict(
        type='HeatmapHead',
        in_channels=17,
        out_channels=17,
        deconv_out_channels=None,
        loss=dict(type='KeypointMSELoss1', use_target_weight=True),
        decoder=codec),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=True,
        modulate_kernel=11))

dataset_type = 'AP10KDataset'
data_mode = 'topdown'
data_root = '/root/mmpose/data/ap10k/'

# pipelines
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    # dict(
    #     type='RandomBBoxTransform', scale_factor=[0.6, 1.4], rotate_factor=70),
    dict(
        type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=(256, 256),use_udp=True),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=1.0), 
                # p=0.5), 
           
        ]),
    dict(type='GenerateTarget', encoder=codec),  # 生成 heatmaps
    dict(type='PackPoseInputs',meta_keys=('img_id', 'img_path', 'bbox','bbox_score','num_keypoints','keypoints','keypoints_visible','area','iscrowd','segmentation','id','category_id','raw_ann_info','instance_text_prompt','input_size','input_center', 'input_scale','flip', 'flip_direction', 'flip_indices', 
                   'skeleton_links'))
]
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(256, 256),use_udp=True),
    dict(type='PackPoseInputs',meta_keys=('img_id', 'img_path', 'bbox','bbox_score','num_keypoints','keypoints','keypoints_visible','area','iscrowd','segmentation','id','category_id','raw_ann_info','instance_text_prompt','input_size','input_center', 'input_scale','flip', 'flip_direction', 'flip_indices', 
                   'skeleton_links'))
]

# data loaders
train_dataloader = dict(
    # batch_size=32,
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/blip2train.json',
        data_prefix=dict(img='data/'),
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/blip2val.json',
        data_prefix=dict(img='data/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/blip2val.json',
        data_prefix=dict(img='data/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))

# evaluators
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/blip2val.json')
test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/blip2val.json')
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        priority=49)
]
