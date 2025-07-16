import torch
import torch.nn as nn
import numpy as np


class OptimalTransport:
    """
    最优传输算法实现
    从L2RM项目中移植并适配到RDE项目
    """
    
    def __init__(self, reg=0.1, max_iter=1000):
        self.reg = reg
        self.max_iter = max_iter
    
    def sinkhorn_log_domain(self, p, q, C, Mask=None, reg=0.1, niter=1000):
        """
        Sinkhorn算法在对数域中的实现
        Args:
            p: 源分布
            q: 目标分布
            C: 代价矩阵
            Mask: 掩码矩阵
            reg: 正则化参数
            niter: 迭代次数
        Returns:
            最优传输矩阵
        """
        def M(u, v):
            """Modified cost for logarithmic updates"""
            return (-C + u.unsqueeze(1) + v.unsqueeze(0)) / reg
        
        def lse(A):
            """Log-sum-exp"""
            return torch.logsumexp(A, dim=1, keepdim=True)
        
        u = torch.zeros_like(p)
        v = torch.zeros_like(q)
        
        for _ in range(niter):
            u = reg * (torch.log(p) - lse(M(u, v)).squeeze()) + u
            v = reg * (torch.log(q) - lse(M(u, v).t()).squeeze()) + v
        
        pi = torch.exp(M(u, v))
        
        if Mask is not None:
            pi = pi * Mask
        
        return pi
    
    def partial_ot(self, p, q, C, s, xi=None, A=None, reg=0.01):
        """
        偏最优传输算法
        Args:
            p: 源分布
            q: 目标分布
            C: 代价矩阵
            s: 传输质量
            xi: 边际松弛参数
            A: 约束矩阵
            reg: 正则化参数
        Returns:
            偏最优传输矩阵
        """
        if xi is None:
            xi = 1.0 / C.shape[0]
        
        C_extended = torch.zeros(C.shape[0] + 1, C.shape[1] + 1)
        C_extended[:-1, :-1] = C
        C_extended[-1, :-1] = torch.max(C) * xi
        C_extended[:-1, -1] = torch.max(C) * xi
        C_extended[-1, -1] = 0
        
        p_extended = torch.zeros(C.shape[0] + 1)
        p_extended[:-1] = p * s
        p_extended[-1] = 1 - s
        
        q_extended = torch.zeros(C.shape[1] + 1)
        q_extended[:-1] = q * s
        q_extended[-1] = 1 - s
        
        pi_extended = self.sinkhorn_log_domain(p_extended, q_extended, C_extended, reg=reg)
        
        pi = pi_extended[:-1, :-1]
        
        return pi.numpy()
    
    def compute_transport_matrix(self, similarity_matrix, p=None, q=None, s=0.5, reg=0.01):
        """
        计算传输矩阵
        Args:
            similarity_matrix: 相似度矩阵
            p: 源分布（如果为None则使用均匀分布）
            q: 目标分布（如果为None则使用均匀分布）
            s: 传输质量
            reg: 正则化参数
        Returns:
            传输矩阵
        """
        cost_matrix = -similarity_matrix
        
        if p is None:
            p = torch.ones(cost_matrix.shape[0]) / cost_matrix.shape[0]
        if q is None:
            q = torch.ones(cost_matrix.shape[1]) / cost_matrix.shape[1]
        
        # 计算偏最优传输
        transport_matrix = self.partial_ot(p, q, cost_matrix, s, reg=reg)
        
        return torch.tensor(transport_matrix)


def create_supervision_matrix(queue_length, batch_size, select_indices, key_img_indices):
    """
    创建监督矩阵
    Args:
        queue_length: 队列长度
        batch_size: 批次大小
        select_indices: 选择的样本索引
        key_img_indices: 关键图像索引
    Returns:
        监督矩阵
    """
    supervision = torch.zeros(queue_length, batch_size)
    for i, j in zip(key_img_indices, select_indices):
        supervision[i, j] = 1
    return supervision 