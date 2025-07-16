#!/bin/bash
root_dir=./data
tau=0.015 
margin=0.1
noisy_rate=0.5  #0.0 0.2 0.5 0.8
select_ratio=0.3
loss=CCL  # 改为使用CCL损失
ccl_method=log  # CCL方法：log, tan, abs, exp, gce, infoNCE
ccl_q=0.5  # GCE损失的参数
ccl_ratio=0.3  # 负样本选择比例
DATASET_NAME=RSTPReid
# CUHK-PEDES ICFG-PEDES RSTPReid

noisy_file=./noiseindex/${DATASET_NAME}_${noisy_rate}.npy
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
    --loss_names ${loss}+sr${select_ratio} \
    --ccl_method $ccl_method \
    --ccl_q $ccl_q \
    --ccl_ratio $ccl_ratio 
 