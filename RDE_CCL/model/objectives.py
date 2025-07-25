import math
import torch
import torch.nn as nn
import torch.nn.functional as F

 
def compute_sdm_per(scores, pid, logit_scale, epsilon=1e-8):
    """
    Similarity Distribution Matching
    """
    batch_size = scores.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    t2i_cosine_theta = scores
    i2t_cosine_theta = t2i_cosine_theta.t()

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta

    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    loss = torch.sum(i2t_loss, dim=1) + torch.sum(t2i_loss, dim=1)

    return loss

def compute_TRL_per(scores, pid, margin = 0.2, tau=0.02):       
    batch_size = scores.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float().cuda()
    mask = 1 - labels

    alpha_1 =((scores/tau).exp()* labels / ((scores/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()
    alpha_2 = ((scores.t()/tau).exp()* labels / ((scores.t()/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()

    pos_1 = (alpha_1 * scores).sum(1)
    pos_2 = (alpha_2 * scores.t()).sum(1)

    neg_1 = (mask*scores).max(1)[0]
    neg_2 = (mask*scores.t()).max(1)[0]

    cost_1 = (margin + neg_1 - pos_1).clamp(min=0)
    cost_2 = (margin + neg_2 - pos_2).clamp(min=0)
    return cost_1 + cost_2

 
def compute_InfoNCE_per(scores, logit_scale):
    
    # cosine similarity as logits
    logits_per_image = logit_scale * scores
    logits_per_text = logits_per_image.t()

    p1 = F.softmax(logits_per_image, dim=1)
    p2 = F.softmax(logits_per_text, dim=1)

    loss = (- p1.diag().log() - p2.diag().log())/2    
    return loss

def compute_TAL_per(scores, pid, tau, margin):
    batch_size = scores.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float().cuda()
    mask = 1 - labels

    alpha_i2t =((scores/tau).exp()* labels / ((scores/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()
    alpha_t2i = ((scores.t()/tau).exp()* labels / ((scores.t()/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()

    loss = (-  (alpha_i2t*scores).sum(1) + tau * ((scores / tau).exp() * mask).sum(1).clamp(max=10e35).log() + margin).clamp(min=0)  \
        +  (-  (alpha_t2i*scores.t()).sum(1) + tau * ((scores.t() / tau).exp() * mask).sum(1).clamp(max=10e35).log() + margin).clamp(min=0)
    
    return loss 

def compute_CCL_per(scores, pid, tau=0.05, method='log', q=0.5, ratio=0):
    """
    Compute CCL (Curriculum Contrastive Learning) loss
    Args:
        scores: similarity matrix
        pid: person IDs
        tau: temperature parameter
        method: loss function type ('log', 'tan', 'abs', 'exp', 'gce', 'infoNCE')
        q: parameter for GCE loss
        ratio: ratio for selecting negative samples
    """
    eps = 1e-10
    scores = (scores / tau).exp()
    i2t = scores / (scores.sum(1, keepdim=True) + eps)
    t2i = scores.t() / (scores.t().sum(1, keepdim=True) + eps)

    # 生成随机掩码用于选择负样本
    randn = torch.rand_like(scores)
    eye = torch.eye(scores.shape[0]).cuda()
    randn[eye > 0] = randn.min(dim=1)[0] - 1
    n = scores.shape[0]
    num = n - 1 if ratio <= 0 or ratio >= 1 else int(ratio * n)
    V, K = randn.topk(num, dim=1)
    mask = torch.zeros_like(scores)
    mask[torch.arange(n).reshape([-1, 1]).cuda(), K] = 1.

    # 根据method选择不同的损失函数
    if method == 'log':
        criterion = lambda x: -((1. - x + eps).log() * mask).sum(1)
    elif method == 'tan':
        criterion = lambda x: (x.tan() * mask).sum(1)
    elif method == 'abs':
        criterion = lambda x: (x * mask).sum(1)
    elif method == 'exp':
        criterion = lambda x: ((-(1. - x)).exp() * mask).sum(1)
    elif method == 'gce':
        criterion = lambda x: ((1. - (1. - x + eps) ** q) / q * mask).sum(1)
    elif method == 'infoNCE':
        criterion = lambda x: -x.diag()
    else:
        raise Exception('Unknown Loss Function!')
    
    loss = criterion(i2t) + criterion(t2i)
    return loss, scores.diag()

def compute_rbs(i_feats, t_feats, i_tse_f, t_tse_f, pid, label_hat=None, tau=0.02, margin=0.1, loss_type='TAL', logit_scale=50):

    loss_bgm, _ = compute_per_loss(i_feats, t_feats, pid, tau, margin, loss_type, logit_scale)
    loss_tse, _ = compute_per_loss(i_tse_f, t_tse_f, pid, tau, margin, loss_type, logit_scale)

    loss_bgm = (label_hat*loss_bgm).sum()
    loss_tse = (label_hat*loss_tse).sum()
    
    if loss_type in ['TAL','TRL']:
        return loss_bgm, loss_tse
    else:
        return loss_bgm/label_hat.sum(), loss_tse/label_hat.sum() # mean

def compute_per_loss(image_features, text_features, pid, tau=0.02, margin=0.2, loss_type='TAL', logit_scale=50):
    
    # normalized features
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)
    scores = text_norm @ image_norm.t()

    if 'TAL' in loss_type:
        per_loss = compute_TAL_per(scores, pid, tau, margin=margin)
    elif 'TRL' in loss_type:
        per_loss = compute_TRL_per(scores, pid, tau=tau, margin=margin)
    elif 'InfoNCE' in loss_type:
        per_loss = compute_InfoNCE_per(scores, logit_scale)
    elif 'SDM' in loss_type:
        per_loss = compute_sdm_per(scores, pid, logit_scale)
    elif 'CCL' in loss_type:
        per_loss, _ = compute_CCL_per(scores, pid, tau=tau)  # 使用默认的method='log', q=0.5, ratio=0
    else:
        exit()

    return per_loss, scores.diag()



