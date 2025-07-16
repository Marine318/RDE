import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward_Network(nn.Module):
    """
    代价函数网络，用于学习图像-文本对的传输代价
    从L2RM项目中移植并适配到RDE项目
    """
    def __init__(self, queue_length=128):
        super(FeedForward_Network, self).__init__()
        # 重新设计：使用逐元素变换而不是线性层
        # 这样可以处理任意大小的相似度矩阵
        self.queue_length = queue_length
        self.transform = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        self.init_weights()

    def forward(self, x):
        """
        前向传播
        Args:
            x: 相似度矩阵 [queue_length, batch_size]
        Returns:
            处理后的代价矩阵 [queue_length, batch_size]
        """
        x = x.to(dtype=next(self.parameters()).dtype)
        original_shape = x.shape
        x = x.view(-1, 1)
        x = self.transform(x)
        x = x.view(original_shape)
        x = 1.0 - x
        
        return x

    def init_weights(self):
        """
        初始化权重
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    @property
    def size(self):
        """为了兼容性，返回queue_length"""
        return self.queue_length


class CostFunctionLoss(nn.Module):
    """
    代价函数损失计算模块
    """
    def __init__(self):
        super(CostFunctionLoss, self).__init__()

    def forward(self, cost_matrix, supervision_matrix):
        """
        计算代价函数损失
        Args:
            cost_matrix: 代价矩阵 [queue_length, batch_size]
            supervision_matrix: 监督矩阵 [queue_length, batch_size]
        Returns:
            loss: 损失值
        """
        loss = torch.sum(supervision_matrix * cost_matrix, dim=(-2, -1))
        return loss 