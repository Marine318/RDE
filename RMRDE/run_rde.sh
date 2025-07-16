#!/bin/bash
root_dir=./data
tau=0.015 
margin=0.1
noisy_rate=0.5  #0.0 0.2 0.5 0.8
select_ratio=0.3
loss=TAL
DATASET_NAME=RSTPReid
# CUHK-PEDES ICFG-PEDES RSTPReid

# L2RM集成参数配置
USE_L2RM=true  # 设置为false可以关闭L2RM功能
lr_cost=1e-4
queue_length=128
rho=0.5
reg=0.01
l2rm_loss_weight=1.0
noise_ratio=0.0

# 不同噪声水平的推荐配置：
# 低噪声 (noisy_rate=0.2): l2rm_loss_weight=0.3, queue_length=128
# 中噪声 (noisy_rate=0.5): l2rm_loss_weight=0.8, queue_length=256
# 高噪声 (noisy_rate=0.8): l2rm_loss_weight=1.5, queue_length=256, rho=0.3

noisy_file=./noiseindex/${DATASET_NAME}_${noisy_rate}.npy

# 根据噪声水平自动调整L2RM参数
if (( $(echo "$noisy_rate >= 0.8" | bc -l) )); then
    echo "🔥 检测到高噪声环境 (noisy_rate=$noisy_rate)"
    l2rm_loss_weight=1.5
    queue_length=256
    rho=0.3
elif (( $(echo "$noisy_rate >= 0.5" | bc -l) )); then
    echo "⚠️  检测到中等噪声环境 (noisy_rate=$noisy_rate)"
    l2rm_loss_weight=0.8
    queue_length=256
elif (( $(echo "$noisy_rate >= 0.2" | bc -l) )); then
    echo "⚡ 检测到低噪声环境 (noisy_rate=$noisy_rate)"
    l2rm_loss_weight=0.3
    queue_length=128
else
    echo "✨ 检测到干净环境 (noisy_rate=$noisy_rate)"
    l2rm_loss_weight=0.1  # 即使是干净环境也可以轻微启用L2RM
    queue_length=64
fi

# 构建L2RM参数
L2RM_ARGS=""
if [ "$USE_L2RM" = true ]; then
    L2RM_ARGS="--use_cost_function --use_l2rm_noisy_loss --lr_cost $lr_cost --queue_length $queue_length --rho $rho --reg $reg --l2rm_loss_weight $l2rm_loss_weight --noise_ratio $noise_ratio"
    echo ""
    echo "🚀 L2RM集成已启用"
    echo "   - 代价函数学习率: $lr_cost"
    echo "   - 队列长度: $queue_length"
    echo "   - L2RM损失权重: $l2rm_loss_weight"
    echo "   - 传输参数: rho=$rho, reg=$reg"
    echo ""
else
    echo ""
    echo "❌ L2RM集成已关闭"
    echo ""
fi

echo "📊 实验配置总结:"
echo "   🔵 数据集: $DATASET_NAME"
echo "   🔵 噪声率: $noisy_rate"
echo "   🔵 损失函数: $loss"
echo "   🔵 批次大小: 64"
echo "   🔵 选择比例: $select_ratio"
echo "   🔵 温度参数: $tau"
echo "   🔵 边距参数: $margin"
if [ "$USE_L2RM" = true ]; then
echo "   🟢 L2RM集成: 启用 (权重: $l2rm_loss_weight, 队列: $queue_length)"
else
echo "   🔴 L2RM集成: 关闭"
fi
echo "========================================"

CUDA_VISIBLE_DEVICES=0 \
    python train.py \
    --noisy_rate $noisy_rate \
    --noisy_file $noisy_file \
    --name RDE \
    --img_aug \
    --txt_aug \
    --batch_size 64 \
    --select_ratio $select_ratio \
    --tau $tau \
    --root_dir $root_dir \
    --output_dir run_logs \
    --margin $margin \
    --dataset_name $DATASET_NAME \
    --loss_names ${loss}+sr${select_ratio}_tau${tau}_margin${margin}_n${noisy_rate}  \
    --num_epoch 60 \
    $L2RM_ARGS
 