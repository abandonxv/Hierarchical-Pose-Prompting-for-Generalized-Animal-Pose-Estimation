#!/usr/bin/env bash
# Copyright (c) OpenMMLab. All rights reserved.

CONFIG=$1
GPU_ID=${2:-5}  # 默认为GPU 7
RESUME_PATH=${3:-}  # 添加checkpoint恢复路径参数，默认为空

# 设置CUDA_VISIBLE_DEVICES环境变量限定使用的GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

# 添加恢复训练逻辑
if [ -n "$RESUME_PATH" ]; then
    RESUME_ARG="--resume $RESUME_PATH"
    echo "Resuming training from checkpoint: $RESUME_PATH"
else
    RESUME_ARG=""
    echo "Starting new training session"
fi

# 使用单GPU运行训练脚本
python $(dirname "$0")/train.py \
    $CONFIG \
    --launcher none \
    $RESUME_ARG \
    ${@:4}  # 保留其他参数，从第4个参数开始

