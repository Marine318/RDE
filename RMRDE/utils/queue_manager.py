import torch
import random
import numpy as np
from typing import List, Tuple, Optional


class RawImageQueue:
    """
    存储原始图像数据的队列，用于L2RM代价函数训练
    """
    def __init__(self, queue_length: int, image_shape: Tuple[int, int, int] = (3, 384, 128)):
        """
        初始化原始图像队列
        Args:
            queue_length: 队列长度
            image_shape: 图像形状 (C, H, W)
        """
        self.queue_length = queue_length
        self.image_shape = image_shape
        self.queue = None
        self.is_initialized = False
        
    def initialize_queue(self, images: torch.Tensor):
        """
        初始化队列
        Args:
            images: 图像数据 [N, C, H, W]
        """
        num_images = images.shape[0]
        
        if num_images < self.queue_length:
            # 如果图像数量不足，进行重复采样
            indices = np.random.choice(num_images, self.queue_length, replace=True)
            selected_images = images[indices]
        else:
            # 随机选择queue_length个图像
            indices = np.random.choice(num_images, self.queue_length, replace=False)
            selected_images = images[indices]
        
        self.queue = selected_images.clone().detach()
        self.is_initialized = True
        
    def update_queue(self, new_images: torch.Tensor, batch_size: int):
        """
        更新队列（FIFO方式）
        Args:
            new_images: 新的图像数据 [batch_size, C, H, W]
            batch_size: 批次大小
        """
        if not self.is_initialized:
            raise ValueError("Queue not initialized. Call initialize_queue first.")
        
        batch_size = min(batch_size, new_images.shape[0])
        
        if batch_size >= self.queue_length:
            # 如果批次大小大于等于队列长度，直接替换整个队列
            self.queue = new_images[:self.queue_length].clone().detach()
        else:
            # 移动队列并添加新图像
            self.queue[batch_size:] = self.queue[:-batch_size].clone()
            self.queue[:batch_size] = new_images[:batch_size].clone().detach()
    
    def get_queue(self) -> torch.Tensor:
        """
        获取当前队列
        Returns:
            当前队列 [queue_length, C, H, W]
        """
        if not self.is_initialized or self.queue is None:
            raise ValueError("Queue not initialized. Call initialize_queue first.")
        return self.queue
    
    def get_queue_for_loss(self) -> torch.Tensor:
        """
        获取用于损失计算的队列
        Returns:
            当前队列 [queue_length, C, H, W]
        """
        return self.get_queue()
    
    def update_queue_l2rm_style(self, noisy_images: torch.Tensor):
        """
        L2RM风格的队列更新
        Args:
            noisy_images: 噪声图像数据 [batch_size, C, H, W]
        """
        if not self.is_initialized:
            raise ValueError("Queue not initialized. Call initialize_queue first.")
        
        batch_size = min(noisy_images.shape[0], self.queue_length)
        
        if batch_size >= self.queue_length:
            # 如果批次大小大于等于队列长度，直接替换整个队列
            self.queue = noisy_images[:self.queue_length].clone().detach()
        else:
            # 移动队列并添加新图像
            self.queue[batch_size:] = self.queue[:-batch_size].clone()
            self.queue[:batch_size] = noisy_images[:batch_size].clone().detach()
    
    def update_queue_after_training(self, batch_images: torch.Tensor):
        """
        训练后更新队列
        Args:
            batch_images: 批次图像数据 [batch_size, C, H, W]
        """
        batch_size = min(batch_images.shape[0], self.queue_length)
        
        if batch_size >= self.queue_length:
            # 如果批次大小大于等于队列长度，直接替换整个队列
            self.queue = batch_images[:self.queue_length].clone().detach()
        else:
            # 移动队列并添加新图像
            self.queue[batch_size:] = self.queue[:-batch_size].clone()
            self.queue[:batch_size] = batch_images[:batch_size].clone().detach()

class ImageQueue:
    """
    图像队列管理器
    用于维护历史图像特征队列，支持代价函数训练
    """
    
    def __init__(self, queue_length: int, embed_dim: int = 512):
        """
        初始化图像队列
        Args:
            queue_length: 队列长度
            embed_dim: 图像特征维度（此处保留兼容性，实际存储原始图像）
        """
        self.queue_length = queue_length
        self.embed_dim = embed_dim  # 保留参数兼容性
        self.queue = None
        self.is_initialized = False
    
    def initialize_queue(self, images: torch.Tensor):
        """
        初始化队列
        Args:
            images: 原始图像数据集合 [N, C, H, W]
        """
        num_images = images.shape[0]
        if num_images < self.queue_length:
            # 如果图像数量不足，重复采样
            indices = np.random.choice(num_images, self.queue_length, replace=True)
        else:
            # 随机选择图像
            indices = np.random.choice(num_images, self.queue_length, replace=False)
        
        self.queue = images[indices].clone().detach()
        self.is_initialized = True
    
    def update_queue(self, new_images: torch.Tensor, batch_size: int):
        """
        更新队列
        Args:
            new_images: 新的图像特征 [batch_size, embed_dim]
            batch_size: 批次大小
        """
        if not self.is_initialized or self.queue is None:
            raise ValueError("Queue not initialized. Call initialize_queue first.")
        
        # 队列循环更新
        if batch_size >= self.queue_length:
            # 如果批次大小大于队列长度，直接替换
            self.queue = new_images[:self.queue_length].clone().detach()
        else:
            # 移动队列并添加新图像
            self.queue[batch_size:] = self.queue[:-batch_size].clone()
            self.queue[:batch_size] = new_images.clone().detach()
    
    def get_queue(self) -> torch.Tensor:
        """
        获取当前队列
        Returns:
            当前队列 [queue_length, embed_dim]
        """
        if not self.is_initialized or self.queue is None:
            raise ValueError("Queue not initialized. Call initialize_queue first.")
        return self.queue
    
    def create_key_queue(self, clean_images: torch.Tensor, batch_size: int) -> Tuple[torch.Tensor, List[int], List[int]]:
        """
        创建关键队列用于代价函数训练
        Args:
            clean_images: 干净图像特征 [batch_size, embed_dim]
            batch_size: 批次大小
        Returns:
            key_queue: 关键队列 [queue_length, embed_dim]
            select_indices: 选择的样本索引
            key_img_indices: 关键图像索引
        """
        if not self.is_initialized or self.queue is None:
            raise ValueError("Queue not initialized. Call initialize_queue first.")
        
        # 复制当前队列
        key_queue = self.queue.clone().detach()
        
        # 确定关键点数量（批次大小的10%）
        key_point_num = max(1, int(batch_size * 0.1))
        
        # 随机选择样本索引
        select_indices = random.sample(range(batch_size), key_point_num)
        
        # 随机选择队列中的位置
        key_img_indices = random.sample(range(self.queue_length), key_point_num)
        
        # 用选择的干净图像替换队列中的图像
        for i, j in zip(key_img_indices, select_indices):
            key_queue[i] = clean_images[j]
        
        return key_queue, select_indices, key_img_indices
    
    def update_queue_l2rm_style(self, noisy_images: torch.Tensor):
        """
        L2RM风格的队列更新：用噪声图像更新队列
        Args:
            noisy_images: 噪声图像 [noisy_batch_size, C, H, W]
        """
        if not self.is_initialized or self.queue is None:
            raise ValueError("Queue not initialized. Call initialize_queue first.")
        
        if noisy_images.shape[0] > 0:
            batch_size = min(noisy_images.shape[0], self.queue_length)
            
            # L2RM风格的循环队列更新
            self.queue[batch_size:] = self.queue[:-batch_size].clone()
            self.queue[:batch_size] = noisy_images[:batch_size].clone()
    
    def update_queue_after_training(self, batch_images: torch.Tensor):
        """
        在主模型训练后更新队列（遵循L2RM的更新时机）
        Args:
            batch_images: 当前批次的图像数据 [batch_size, C, H, W]
        """
        if not self.is_initialized or self.queue is None:
            raise ValueError("Queue not initialized. Call initialize_queue first.")
        
        if batch_images.shape[0] > 0:
            batch_size = min(batch_images.shape[0], self.queue_length)
            
            # 将队列向后移动batch_size位
            self.queue[batch_size:] = self.queue[:-batch_size].clone()
            
            # 新图像填充队列前部
            self.queue[:batch_size] = batch_images[:batch_size].clone()
    
    def get_queue_for_loss(self) -> torch.Tensor:
        """
        获取当前队列用于损失计算
        Returns:
            queue: 当前图像队列 [queue_length, C, H, W]
        """
        if not self.is_initialized or self.queue is None:
            raise ValueError("Queue not initialized. Call initialize_queue first.")
        return self.queue
    
    def reset_queue(self):
        """
        重置队列
        """
        self.queue = None
        self.is_initialized = False
    
    def get_queue_stats(self) -> dict:
        """
        获取队列统计信息
        Returns:
            队列统计信息字典
        """
        if not self.is_initialized or self.queue is None:
            return {"initialized": False}
        
        return {
            "initialized": True,
            "queue_length": self.queue_length,
            "embed_dim": self.embed_dim,
            "queue_shape": self.queue.shape,
            "queue_mean": self.queue.mean().item(),
            "queue_std": self.queue.std().item()
        }


class QueueManager:
    """
    多队列管理器
    可以管理多个不同类型的队列
    """
    
    def __init__(self):
        self.queues = {}
    
    def create_queue(self, name: str, queue_length: int, embed_dim: int = 512) -> ImageQueue:
        """
        创建新队列
        Args:
            name: 队列名称
            queue_length: 队列长度
            embed_dim: 特征维度
        Returns:
            创建的队列对象
        """
        queue = ImageQueue(queue_length, embed_dim)
        self.queues[name] = queue
        return queue
    
    def get_queue(self, name: str) -> Optional[ImageQueue]:
        """
        获取队列
        Args:
            name: 队列名称
        Returns:
            队列对象或None
        """
        return self.queues.get(name)
    
    def remove_queue(self, name: str):
        """
        移除队列
        Args:
            name: 队列名称
        """
        if name in self.queues:
            del self.queues[name]
    
    def list_queues(self) -> List[str]:
        """
        列出所有队列名称
        Returns:
            队列名称列表
        """
        return list(self.queues.keys())
    
    def reset_all_queues(self):
        """
        重置所有队列
        """
        for queue in self.queues.values():
            queue.reset_queue()
    
    def get_all_stats(self) -> dict:
        """
        获取所有队列的统计信息
        Returns:
            所有队列的统计信息
        """
        stats = {}
        for name, queue in self.queues.items():
            stats[name] = queue.get_queue_stats()
        return stats 