import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SymmetricKLLoss(nn.Module):
    """
    对称KL散度损失，从L2RM项目移植
    实现sym_KL_Divergence功能
    """
    def __init__(self, tau=0.1):
        super(SymmetricKLLoss, self).__init__()
        self.tau = tau
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(self, scores, soft_labels):
        """
        计算对称KL散度损失
        Args:
            scores: 相似度矩阵 [queue_length, batch_size]
            soft_labels: 最优传输矩阵（软标签）[queue_length, batch_size]
        Returns:
            loss: 对称KL散度损失
        """
        eps = 1e-10
        scores = (scores / self.tau).exp()
        
        i2t = scores / (scores.sum(1, keepdim=True) + eps)
        t2i = scores.t() / (scores.t().sum(1, keepdim=True) + eps)

        soft_labels = soft_labels.clip(min=eps)
        normalized_labels_i2t = soft_labels / soft_labels.sum(dim=1, keepdim=True)
        normalized_labels_t2i = soft_labels.t() / soft_labels.t().sum(dim=1, keepdim=True)

        cost_i2t = self.kl_loss(i2t.log(), normalized_labels_i2t)
        cost_t2i = self.kl_loss(t2i.log(), normalized_labels_t2i)

        cost_i2t_r = self.kl_loss(normalized_labels_i2t.log(), i2t)
        cost_t2i_r = self.kl_loss(normalized_labels_t2i.log(), t2i)

        SKL_loss = 0.5 * (cost_i2t + cost_t2i + cost_i2t_r + cost_t2i_r)
        return SKL_loss


class PartialOptimalTransport(nn.Module):
    """
    部分最优传输算法，从L2RM项目移植
    实现partial_ot + sinkhorn_log_domain功能
    """
    def __init__(self, reg=0.01, noise_ratio=0.0):
        super(PartialOptimalTransport, self).__init__()
        self.reg = reg
        self.noise_ratio = noise_ratio

    def sinkhorn_log_domain(self, p, q, C, Mask=None, reg=0.1, niter=1000):
        """
        Sinkhorn算法的对数域实现
        Args:
            p: 行边际分布
            q: 列边际分布
            C: 代价矩阵
            Mask: 掩码矩阵
            reg: 正则化参数
            niter: 迭代次数
        Returns:
            pi: 最优传输矩阵
        """
        def M(u, v):
            return (-C + u.reshape(-1, 1) + v.reshape(1, -1)) / reg

        def lse(A):
            return torch.logsumexp(A, dim=1, keepdim=True)

        u, v = torch.zeros_like(p), torch.zeros_like(q)
        
        for _ in range(niter):
            u = reg * (torch.log(p + 1e-8) - lse(M(u, v)).squeeze())
            v = reg * (torch.log(q + 1e-8) - lse(M(u, v).t()).squeeze())
        
        pi = torch.exp(M(u, v))
        
        if Mask is not None:
            pi = pi * Mask.float()
        
        return pi

    def partial_ot(self, p, q, C, s, xi=None, A=None, reg=0.01):
        """
        部分最优传输算法
        Args:
            p: 行边际分布
            q: 列边际分布
            C: 代价矩阵
            s: 传输质量
            xi: 惩罚参数
            A: 最大代价
            reg: 正则化参数
        Returns:
            pi: 部分最优传输矩阵
        """
        if A is None:
            A = C.max()
        if xi is None:
            xi = 1e2 * C.max()
        
        I = list(range(C.shape[1]))
        J = list(range(C.shape[1]))
        
        C_ = torch.cat((C, xi * torch.ones(C.size(0), 1)), dim=1)
        C_ = torch.cat((C_, xi * torch.ones(1, C_.size(1))), dim=0)
        C_[-1, -1] = 2 * xi + A
        
        M = torch.ones_like(C, dtype=torch.float32)
        M[I, :] = 1
        M[:, J] = 1
        M[I, J] = 1 if self.noise_ratio == 0 else 0
        
        a = torch.ones(M.size(0), 1, dtype=torch.float32)
        b = torch.ones(M.size(1) + 1, 1, dtype=torch.float32)
        M_ = torch.cat((M, a), dim=1)
        M_ = torch.cat((M_, b.t()), dim=0)
        
        p_ = torch.cat((p, (torch.sum(p) - s) * torch.tensor([1])))
        q_ = torch.cat((q, (torch.sum(q) - s) * torch.tensor([1])))
        
        pi_ = self.sinkhorn_log_domain(p_, q_, C_, M_, reg=reg)
        pi_ = pi_[:-1, :-1]
        
        return pi_

    def compute_transport_matrix(self, cost_matrix, rho, reg):
        """
        计算最优传输矩阵的封装接口
        Args:
            cost_matrix: 代价矩阵 [queue_length, batch_size]
            rho: 传输质量参数
            reg: 正则化参数
        Returns:
            transport_matrix: 最优传输矩阵
        """
        cost_matrix = cost_matrix.float().cpu().clip(1e-10)
        
        p = np.ones(cost_matrix.shape[0]) / cost_matrix.shape[0]
        q = np.ones(cost_matrix.shape[1]) / cost_matrix.shape[1]
        
        transport_matrix = self.partial_ot(
            torch.tensor(p), torch.tensor(q), cost_matrix, 
            s=rho, reg=reg
        )
        
        return transport_matrix.cuda()


class L2RMNoisyLoss(nn.Module):
    """
    L2RM噪声损失的完整实现
    结合代价函数、最优传输和对称KL散度
    """
    def __init__(self, tau=0.1, reg=0.01, rho=0.5):
        super(L2RMNoisyLoss, self).__init__()
        self.symmetric_kl_loss = SymmetricKLLoss(tau=tau)
        self.optimal_transport = PartialOptimalTransport(reg=reg)
        self.rho = rho
        self.reg = reg

    def forward(self, similarity_matrix, cost_function):
        """
        计算L2RM风格的噪声损失
        Args:
            similarity_matrix: 相似度矩阵 [queue_length, batch_size]
            cost_function: 代价函数网络
        Returns:
            loss: L2RM噪声损失
        """
        similarity_matrix = similarity_matrix.float()
        
        cost_matrix = cost_function(similarity_matrix)
        
        transport_matrix = self.optimal_transport.compute_transport_matrix(
            cost_matrix, self.rho, self.reg
        )
        
        loss = self.symmetric_kl_loss(similarity_matrix, transport_matrix)
        
        return loss 