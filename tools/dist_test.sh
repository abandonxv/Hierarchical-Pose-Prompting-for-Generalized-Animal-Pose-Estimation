#!/usr/bin/env bash
# Copyright (c) OpenMMLab. All rights reserved.

CONFIG=$1          # 第1个参数：配置文件路径
CHECKPOINT=$2      # 第2个参数：权重文件路径
GPU_ID=${3:-0}     # 第3个参数：GPU编号，默认用0号GPU

# 设置Python路径
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

# 指定使用的GPU
CUDA_VISIBLE_DEVICES=$GPU_ID python $(dirname "$0")/test.py \
    $CONFIG \
    $CHECKPOINT \
    ${@:4}
