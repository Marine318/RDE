#!/usr/bin/env python3
"""
测试队列和相似度矩阵修复的脚本
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.build import build_model
from utils.queue_manager import RawImageQueue

# 创建测试参数
class TestArgs:
    def __init__(self):
        # L2RM相关
        self.use_l2rm_noisy_loss = True
        self.queue_length = 128
        self.batch_size = 64
        self.embed_dim = 512
        self.noise_ratio = 0.2
        self.l2rm_loss_weight = 0.3
        self.tau = 0.015
        self.reg = 0.01
        self.rho = 0.5
        self.lr_cost = 1e-4
        
        # 模型相关
        self.current_task = 'TRL'
        self.loss_names = 'sdm+id+mlm'
        self.select_ratio = 0.3
        self.temperature = 0.02
        self.num_classes = 11003
        self.pretrain_choice = 'ViT-B/16'
        self.img_size = (384, 128)
        self.stride_size = 16
        self.text_length = 77
        self.vocab_size = 49408
        self.margin = 0.1
        
        # 训练相关
        self.optimizer = 'Adam'
        self.lr = 1e-5
        self.bias_lr_factor = 2.0
        self.momentum = 0.9
        self.weight_decay = 4e-5
        self.weight_decay_bias = 0.0
        self.alpha = 0.9
        self.beta = 0.999
        
        # 调度器相关
        self.num_epoch = 60
        self.milestones = [20, 50]
        self.gamma = 0.1
        self.warmup_factor = 0.1
        self.warmup_epochs = 5
        self.warmup_method = "linear"
        self.lrscheduler = "cosine"
        self.target_lr = 0
        self.power = 0.9
        
        # 数据集相关
        self.dataset_name = "CUHK-PEDES"
        self.sampler = "random"
        self.num_instance = 4
        self.root_dir = "/home/qinyang/projects/data"
        self.test_batch_size = 512
        self.num_workers = 8
        self.training = True
        
        # 其他
        self.local_rank = 0
        self.name = "baseline"
        self.output_dir = "logs"
        self.log_period = 100
        self.eval_period = 1
        self.val_dataset = "test"
        self.resume = False
        self.resume_ckpt_file = ""
        self.img_aug = False
        self.txt_aug = False
        self.cmt_depth = 4
        self.masked_token_rate = 0.8
        self.masked_token_unchanged_rate = 0.1
        self.lr_factor = 5.0
        self.noisy_rate = 0.2
        self.noisy_file = ''
        self.use_cost_function = False

def test_queue_and_similarity():
    """测试队列和相似度矩阵计算"""
    print("🔧 Testing queue and similarity matrix fix...")
    
    # 初始化参数和模型
    args = TestArgs()
    model = build_model(args)
    
    # 确保模型使用Float32精度（CPU兼容）
    model = model.float()
    
    print(f"✅ Model built successfully")
    print(f"📋 Cost function size: {model.cost_function.size}")
    print(f"📋 Queue length: {args.queue_length}")
    print(f"📋 Batch size: {args.batch_size}")
    
    # 创建测试数据（确保是Float32）
    queue_images = torch.randn(args.queue_length, 3, 384, 128).float()
    batch_captions = torch.randint(0, 49408, (args.batch_size//4, 77))  # 使用小批次
    
    print(f"📋 Queue images shape: {queue_images.shape}")
    print(f"📋 Batch captions shape: {batch_captions.shape}")
    
    # 测试RawImageQueue
    raw_queue = RawImageQueue(args.queue_length, (3, 384, 128))
    sample_images = torch.randn(100, 3, 384, 128)
    raw_queue.initialize_queue(sample_images)
    
    queue_data = raw_queue.get_queue()
    print(f"✅ RawImageQueue works: {queue_data.shape}")
    
    # 测试相似度矩阵计算
    try:
        similarity_matrix = model.compute_queue_similarity_matrix(queue_images, batch_captions)
        print(f"✅ Similarity matrix shape: {similarity_matrix.shape}")
        print(f"📋 Expected shape: [{args.queue_length}, {batch_captions.shape[0]}]")
        
        # 验证形状是否正确
        expected_shape = (args.queue_length, batch_captions.shape[0])
        if similarity_matrix.shape == expected_shape:
            print("✅ Similarity matrix shape is correct!")
        else:
            print(f"❌ Similarity matrix shape mismatch: {similarity_matrix.shape} vs {expected_shape}")
            return False
            
    except Exception as e:
        print(f"❌ Similarity matrix computation failed: {e}")
        return False
    
    # 测试cost function
    try:
        cost_matrix = model.cost_function(similarity_matrix)
        print(f"✅ Cost matrix shape: {cost_matrix.shape}")
        print(f"📋 Expected shape: [{args.queue_length}, {batch_captions.shape[0]}]")
        
        # 验证形状是否正确
        expected_shape = (args.queue_length, batch_captions.shape[0])
        if cost_matrix.shape == expected_shape:
            print("✅ Cost matrix shape is correct!")
        else:
            print(f"❌ Cost matrix shape mismatch: {cost_matrix.shape} vs {expected_shape}")
            return False
            
    except Exception as e:
        print(f"❌ Cost function computation failed: {e}")
        return False
    
    print("\n🎉 All tests passed! The queue and similarity matrix fix is working!")
    return True

if __name__ == "__main__":
    success = test_queue_and_similarity()
    sys.exit(0 if success else 1) 