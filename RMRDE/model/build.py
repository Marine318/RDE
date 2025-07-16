from model import objectives

from .CrossEmbeddingLayer_tse import TexualEmbeddingLayer, VisualEmbeddingLayer
from .clip_model import build_CLIP_from_openai_pretrained, convert_weights
from .cost_function import FeedForward_Network, CostFunctionLoss
from .optimal_transport import OptimalTransport, create_supervision_matrix
from .l2rm_losses import L2RMNoisyLoss, SymmetricKLLoss, PartialOptimalTransport
import torch
import torch.nn as nn 
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

class RDE(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']

        self.logit_scale = torch.ones([]) * (1 / args.temperature) 
 
        self.visul_emb_layer = VisualEmbeddingLayer(ratio=args.select_ratio)
        self.texual_emb_layer = TexualEmbeddingLayer(ratio=args.select_ratio)
        
        if args.use_l2rm_noisy_loss:
            from .cost_function import FeedForward_Network, CostFunctionLoss
            self.cost_function = FeedForward_Network(args.queue_length)
            self.cost_loss_fn = CostFunctionLoss()
            
            self.l2rm_noisy_loss = L2RMNoisyLoss(
                tau=args.tau,
                reg=args.reg,
                rho=args.rho
            )
            logger.info(f"L2RM cost function initialized with size: {args.queue_length}")
        else:
            self.cost_function = None
            self.cost_loss_fn = None
            self.l2rm_noisy_loss = None
 
        if 'TAL' in self.current_task:
            loss_type = 'TAL'
        elif 'TRL' in self.current_task:
            loss_type = 'TRL'
        elif 'InfoNCE' in self.current_task:
            loss_type = 'InfoNCE'
        else:
            loss_type = 'TAL'
            
        self.loss_type = loss_type

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [loss_names]

    def encode_image(self, image):
        image_features, _ = self.base_model.encode_image(image)
        if len(image_features.shape) == 3:  # [batch, seq_len, embed_dim]
            return image_features[:, 0, :].float()
        return image_features.float()

    def encode_text(self, text):
        text_features, _ = self.base_model.encode_text(text)
        text_features = text_features[torch.arange(text_features.shape[0]), text.argmax(dim=-1)].float()
        return text_features

    def encode_image_tse(self, image):
        image_feats, atten_i, text_feats, atten_t = self.base_model(image, None)
        return self.visul_emb_layer(image_feats, atten_i)

    def encode_text_tse(self, text):
        image_feats, atten_i, text_feats, atten_t = self.base_model(None, text)
        return self.texual_emb_layer(text_feats, text, atten_t)

    def compute_per_loss(self, batch):
        images = batch['images']
        caption_ids = batch['caption_ids']
        image_feats, atten_i, text_feats, atten_t = self.base_model(images, caption_ids)
        i_feats = image_feats[:, 0, :].float()
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()
        
        i_tse_f = self.visul_emb_layer(image_feats, atten_i)
        t_tse_f = self.texual_emb_layer(text_feats, caption_ids, atten_t)
        
        pid = batch['pids']
        
        if len(pid.shape) == 1:
            pid = pid.unsqueeze(1)
        
        lossA, simsA = objectives.compute_per_loss(i_feats, t_feats, pid, 
                                                   tau=self.args.tau, 
                                                   margin=self.args.margin, 
                                                   loss_type=self.loss_type, 
                                                   logit_scale=self.logit_scale)
        lossB, simsB = objectives.compute_per_loss(i_tse_f, t_tse_f, pid,
                                                   tau=self.args.tau, 
                                                   margin=self.args.margin, 
                                                   loss_type=self.loss_type, 
                                                   logit_scale=self.logit_scale)
        
        return lossA.detach().cpu(), lossB.detach().cpu(), simsA, simsB

    def compute_similarity_matrix(self, images, captions, lengths):
        """
        计算相似度矩阵
        Args:
            images: 图像张量 [batch_size, C, H, W] 或 特征 [batch_size, embed_dim]
            captions: 文本张量 [batch_size, seq_len]
            lengths: 文本长度（RDE中不使用）
        Returns:
            similarity_matrix: 相似度矩阵 [batch_size, batch_size]
        """
        if len(images.shape) == 4:  # 原始图像
            image_features, _ = self.base_model.encode_image(images)
            if len(image_features.shape) == 3:  # [batch, seq_len, embed_dim]
                image_features = image_features[:, 0, :].float()
        else: 
            image_features = images
            
        text_features, _ = self.base_model.encode_text(captions)
        text_features = text_features[torch.arange(text_features.shape[0]), captions.argmax(dim=-1)].float()
        
        image_features = l2norm(image_features)
        text_features = l2norm(text_features)
        
        similarity_matrix = torch.matmul(image_features, text_features.t())
        
        return similarity_matrix

    def compute_queue_similarity_matrix(self, queue_images, batch_captions):
        """
        计算队列图像与批次文本之间的相似度矩阵
        Args:
            queue_images: 队列图像张量 [queue_length, C, H, W]
            batch_captions: 批次文本张量 [batch_size, seq_len]
        Returns:
            similarity_matrix: 相似度矩阵 [queue_length, batch_size]
        """
        # 处理队列图像
        if len(queue_images.shape) == 4:  # 原始图像
            image_features, _ = self.base_model.encode_image(queue_images)
            if len(image_features.shape) == 3:  
                image_features = image_features[:, 0, :].float()
        else:  
            image_features = queue_images
            
        text_features, _ = self.base_model.encode_text(batch_captions)
        text_features = text_features[torch.arange(text_features.shape[0]), batch_captions.argmax(dim=-1)].float()
        
        image_features = l2norm(image_features)
        text_features = l2norm(text_features)
        
        similarity_matrix = torch.matmul(image_features, text_features.t())
        
        return similarity_matrix

    def compute_cost_function_loss(self, key_queue, captions, lengths, select_indices, key_img_indices):
        """
        计算代价函数损失
        Args:
            key_queue: 关键队列 [queue_length, embed_dim]
            captions: 文本张量 [batch_size, seq_len]
            lengths: 文本长度
            select_indices: 选择的样本索引
            key_img_indices: 关键图像索引
        Returns:
            cost_loss: 代价函数损失
        """
        if self.cost_function is None or self.cost_loss_fn is None:
            return torch.tensor(0.0)
        
        similarity_matrix = self.compute_similarity_matrix(key_queue, captions, lengths)
        
        cost_matrix = self.cost_function(similarity_matrix)
        
        supervision_matrix = create_supervision_matrix(
            len(key_queue), len(captions), select_indices, key_img_indices
        ).to(cost_matrix.device)
        
        cost_loss = self.cost_loss_fn(cost_matrix, supervision_matrix)
        
        return cost_loss

    def compute_l2rm_noisy_loss(self, noisy_images, noisy_captions, imgs_queue):
        """
        计算L2RM风格的噪声数据损失
        Args:
            noisy_images: 噪声图像 [noisy_batch_size, C, H, W]
            noisy_captions: 噪声文本 [noisy_batch_size, seq_len]
            imgs_queue: 图像队列 [queue_length, C, H, W]
        Returns:
            loss_n: L2RM噪声损失
        """
        if self.l2rm_noisy_loss is None or self.cost_function is None:
            return torch.tensor(0.0)
        
        if noisy_images.shape[0] == 0:
            return torch.tensor(0.0)
            
        similarity_matrix = self.compute_similarity_matrix(
            imgs_queue, noisy_captions, None
        )
        
        loss_n = self.l2rm_noisy_loss(similarity_matrix, self.cost_function)
        
        return loss_n

    def forward(self, batch):
        ret = dict()
        ret.update({'temperature': 1 / self.logit_scale})

        images = batch['images']
        caption_ids = batch['caption_ids']
        image_feats, atten_i, text_feats, atten_t = self.base_model(images, caption_ids)
        i_feats = image_feats[:, 0, :].float()
        # i_feats = image_feats.float() # for CLIP ResNet visual model
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        i_tse_f = self.visul_emb_layer(image_feats, atten_i)
        t_tse_f = self.texual_emb_layer(text_feats, caption_ids, atten_t)
            
        label_hat = batch['label_hat'].to(i_feats.device) 
     
        loss1, loss2 = objectives.compute_rbs(i_feats, t_feats, i_tse_f, t_tse_f, batch['pids'], \
                                              label_hat=label_hat, margin=self.args.margin,tau=self.args.tau,\
                                                loss_type=self.loss_type,logit_scale=self.logit_scale)
        ret.update({'bge_loss':loss1})
        ret.update({'tse_loss':loss2})

        loss_n = torch.tensor(0.0)
        if (self.l2rm_noisy_loss is not None and 
            'imgs_queue' in batch and 
            hasattr(self.args, 'use_l2rm_noisy_loss') and 
            self.args.use_l2rm_noisy_loss):
            noisy_mask = (label_hat == 0)
            
            if noisy_mask.sum() > 0:  # 如果有噪声数据
                noisy_images = images[noisy_mask]
                noisy_captions = caption_ids[noisy_mask]
                imgs_queue = batch['imgs_queue']
                
                loss_n = self.compute_l2rm_noisy_loss(
                    noisy_images, noisy_captions, imgs_queue
                )
                
                if hasattr(self.args, 'l2rm_loss_weight'):
                    loss_n = loss_n * self.args.l2rm_loss_weight
        
        ret.update({'l2rm_noisy_loss': loss_n})
  
        return ret


def build_model(args, num_classes=11003):
    model = RDE(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    return model
