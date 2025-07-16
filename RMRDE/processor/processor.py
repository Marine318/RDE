import logging
import os
import time
import torch
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable
import numpy as np
from matplotlib import pyplot as plt
from pylab import xticks,yticks,np
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture
from utils.queue_manager import ImageQueue, RawImageQueue
import random


################### CODE FOR THE BETA MODEL  ########################

import scipy.stats as stats
def weighted_mean(x, w):
    return np.sum(w * x) / np.sum(w)

def fit_beta_weighted(x, w):
    x_bar = weighted_mean(x, w)
    s2 = weighted_mean((x - x_bar)**2, w)
    alpha = x_bar * ((x_bar * (1 - x_bar)) / s2 - 1)
    beta = alpha * (1 - x_bar) /x_bar
    return alpha, beta

class BetaMixture1D(object):
    def __init__(self, max_iters=10,
                 alphas_init=[1, 2],
                 betas_init=[2, 1],
                 weights_init=[0.5, 0.5]):
        self.alphas = np.array(alphas_init, dtype=np.float64)
        self.betas = np.array(betas_init, dtype=np.float64)
        self.weight = np.array(weights_init, dtype=np.float64)
        self.max_iters = max_iters
        self.lookup = np.zeros(100, dtype=np.float64)
        self.lookup_resolution = 100
        self.lookup_loss = np.zeros(100, dtype=np.float64)
        self.eps_nan = 1e-12

    def likelihood(self, x, y):
        return stats.beta.pdf(x, self.alphas[y], self.betas[y])

    def weighted_likelihood(self, x, y):
        return self.weight[y] * self.likelihood(x, y)

    def probability(self, x):
        return sum(self.weighted_likelihood(x, y) for y in range(2))

    def posterior(self, x, y):
        return self.weighted_likelihood(x, y) / (self.probability(x) + self.eps_nan)

    def responsibilities(self, x):
        r =  np.array([self.weighted_likelihood(x, i) for i in range(2)])
        # there are ~200 samples below that value
        r[r <= self.eps_nan] = self.eps_nan
        r /= r.sum(axis=0)
        return r

    def score_samples(self, x):
        return -np.log(self.probability(x))

    def fit(self, x):
        x = np.copy(x)

        # EM on beta distributions unsable with x == 0 or 1
        eps = 1e-4
        x[x >= 1 - eps] = 1 - eps
        x[x <= eps] = eps

        for i in range(self.max_iters):

            # E-step
            r = self.responsibilities(x)

            # M-step
            self.alphas[0], self.betas[0] = fit_beta_weighted(x, r[0])
            self.alphas[1], self.betas[1] = fit_beta_weighted(x, r[1])
            self.weight = r.sum(axis=1)
            self.weight /= self.weight.sum()

        return self

    def predict(self, x):
        return self.posterior(x, 1) > 0.5

    def create_lookup(self, y):
        x_l = np.linspace(0+self.eps_nan, 1-self.eps_nan, self.lookup_resolution)
        lookup_t = self.posterior(x_l, y)
        lookup_t[np.argmax(lookup_t):] = lookup_t.max()
        self.lookup = lookup_t
        self.lookup_loss = x_l # I do not use this one at the end

    def look_lookup(self, x):
        x_i = x.clone().cpu().numpy()
        x_i = np.array((self.lookup_resolution * x_i).astype(int))
        x_i[x_i < 0] = 0
        x_i[x_i == self.lookup_resolution] = self.lookup_resolution - 1
        return self.lookup[x_i]

    def __str__(self):
        return 'BetaMixture1D(w={}, a={}, b={})'.format(self.weight, self.alphas, self.betas)


def split_prob(prob, threshld):
    if prob.min() > threshld:
        """From https://github.com/XLearning-SCU/2021-NeurIPS-NCR"""
        # If prob are all larger than threshld, i.e. no noisy data, we enforce 1/100 unlabeled data
        print('No estimated noisy data. Enforce the 1/100 data with small probability to be unlabeled.')
        threshld = np.sort(prob)[len(prob)//100]
    pred = (prob > threshld)
    return (pred+0)


def update_cost_function(model, batch, cost_optimizer, image_queue):
    """
    更新代价函数
    Args:
        model: 模型
        batch: 批次数据
        cost_optimizer: 代价函数优化器
        image_queue: 图像队列
    Returns:
        cost_loss: 代价函数损失
    """
    if not hasattr(model, 'cost_function') or model.cost_function is None:
        return torch.tensor(0.0)
    
    clean_indices = batch['label_hat'] == 1
    if clean_indices.sum() == 0:
        return torch.tensor(0.0)
    
    clean_images = batch['images'][clean_indices]
    clean_captions = batch['caption_ids'][clean_indices]
    
    try:
        imgs_queue_key = image_queue.get_queue().clone().detach()
        
        key_point_num = max(1, int(clean_images.shape[0] * 0.1))
        
        select_indices = random.sample(range(clean_images.shape[0]), key_point_num)
        
        key_img_indices = random.sample(range(len(imgs_queue_key)), key_point_num)       

        for i, j in zip(key_img_indices, select_indices):
            imgs_queue_key[i] = clean_images[j]

        similarity_matrix = model.compute_queue_similarity_matrix(imgs_queue_key, clean_captions)
        
        from model.optimal_transport import create_supervision_matrix
        supervision_matrix = create_supervision_matrix(
            len(imgs_queue_key), len(clean_captions), select_indices, key_img_indices
        ).to(similarity_matrix.device)
        
    except Exception:
        return torch.tensor(0.0)
    
    model.cost_function.train()
    cost_matrix = model.cost_function(similarity_matrix)
    cost_loss = model.cost_loss_fn(cost_matrix, supervision_matrix)
    
    if cost_loss.item() > 0:
        cost_optimizer.zero_grad()
        cost_loss.backward()
        cost_optimizer.step()
    
    return cost_loss.detach()


def update_image_queue_after_training(batch, image_queue):
    """
    在训练后更新图像队列（遵循L2RM的时机）
    Args:
        batch: 批次数据
        image_queue: 图像队列
    """
    try:
        noisy_indices = batch['label_hat'] == 0
        if noisy_indices.sum() == 0:
            return
            
        noisy_images = batch['images'][noisy_indices]
        batch_size = len(noisy_images)
        
        if batch_size > 0:
            queue = image_queue.get_queue()
            if batch_size >= len(queue):
                image_queue.queue = noisy_images[:len(queue)].clone().detach()
            else:
                queue[batch_size:] = queue[:-batch_size].clone()
                queue[:batch_size] = noisy_images.clone().detach()
                
    except Exception:
        pass


def get_loss(model, data_loader):
    logger = logging.getLogger("RDE.train")
    model.eval()
    device = "cuda"
    data_size = data_loader.dataset.__len__()
    real_labels = data_loader.dataset.real_correspondences
    lossA, lossB, simsA,simsB = torch.zeros(data_size), torch.zeros(data_size), torch.zeros(data_size),torch.zeros(data_size)
    for i, batch in enumerate(data_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        index = batch['index']
        with torch.no_grad(): 
            la, lb, sa, sb = model.compute_per_loss(batch)
            for b in range(la.size(0)):
                lossA[index[b]]= la[b]
                lossB[index[b]]= lb[b]
                simsA[index[b]]= sa[b]
                simsB[index[b]]= sb[b]
            if i % 100 == 0:
                logger.info(f'compute loss batch {i}')

    losses_A = (lossA-lossA.min())/(lossA.max()-lossA.min())    
    losses_B = (lossB-lossB.min())/(lossB.max()-lossB.min())
    
    input_loss_A = losses_A.reshape(-1,1) 
    input_loss_B = losses_B.reshape(-1,1)
 
    logger.info('\nFitting GMM ...') 
 
    if model.args.noisy_rate > 0.4 or model.args.dataset_name=='RSTPReid':
        # should have a better fit 
        gmm_A = GaussianMixture(n_components=2, max_iter=100, tol=1e-4, reg_covar=1e-6)
        gmm_B = GaussianMixture(n_components=2, max_iter=100, tol=1e-4, reg_covar=1e-6)
    else:
        gmm_A = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
        gmm_B = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)

    gmm_A.fit(input_loss_A.cpu().numpy())
    prob_A = gmm_A.predict_proba(input_loss_A.cpu().numpy())
    prob_A = prob_A[:, gmm_A.means_.argmin()]

    gmm_B.fit(input_loss_B.cpu().numpy())
    prob_B = gmm_B.predict_proba(input_loss_B.cpu().numpy())
    prob_B = prob_B[:, gmm_B.means_.argmin()]
 
 
    pred_A = split_prob(prob_A, 0.5)
    pred_B = split_prob(prob_B, 0.5)
    
    return torch.Tensor(pred_A), torch.Tensor(pred_B)




def do_train(start_epoch, args, model, train_loader, evaluator, optimizer,
             scheduler, checkpointer):

    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logging.getLogger("RDE.train")
    logger.info('start training')

    meters = {
        "loss": AverageMeter(),
        "bge_loss": AverageMeter(),
        "tse_loss": AverageMeter(),
        "id_loss": AverageMeter(),
        "img_acc": AverageMeter(),
        "txt_acc": AverageMeter(),
        "cost_loss": AverageMeter(),
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)

    best_top1 = 0.0
    
    cost_optimizer = None
    image_queue = None
    if args.use_l2rm_noisy_loss and hasattr(model, 'cost_function') and model.cost_function is not None:
        cost_optimizer = torch.optim.Adam(model.cost_function.parameters(), lr=args.lr_cost)
        image_queue = RawImageQueue(args.queue_length, (3, 384, 128))
        logger.info(f"Cost function enabled with queue length: {args.queue_length}")
        
        if hasattr(args, 'use_l2rm_noisy_loss') and args.use_l2rm_noisy_loss:
            logger.info(f"L2RM noisy loss enabled with weight: {getattr(args, 'l2rm_loss_weight', 1.0)}")
        else:
            logger.info("L2RM noisy loss disabled")
    else:
        logger.info("Cost function disabled")
    sims = []
    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()

        model.epoch = epoch
        # data_size = train_loader.dataset.__len__()
        # pred_A, pred_B  =  torch.ones(data_size), torch.ones(data_size)
    
        pred_A, pred_B = get_loss(model, train_loader)
    
        consensus_division = pred_A + pred_B # 0,1,2 
        consensus_division[consensus_division==1] += torch.randint(0, 2, size=(((consensus_division==1)+0).sum(),))
        label_hat = consensus_division.clone()
        label_hat[consensus_division>1] = 1
        label_hat[consensus_division<=1] = 0 
        
        if epoch == start_epoch and image_queue is not None:
            # 收集原始图像数据来初始化队列
            sample_images = []
            for i, batch in enumerate(train_loader):
                if i >= 10:  # 只取前10个batch来初始化
                    break
                batch = {k: v.to(device) for k, v in batch.items()}
                images = batch['images']  # 直接使用原始图像数据
                sample_images.append(images)
            
            if sample_images:
                all_images = torch.cat(sample_images, dim=0)
                image_queue.initialize_queue(all_images)
                logger.info(f"Initialized image queue with {len(all_images)} images")
        
        model.train() 
        for n_iter, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            index = batch['index']
            
            batch['label_hat'] = label_hat[index.cpu()]
            
            cost_loss = torch.tensor(0.0)
            if cost_optimizer is not None and image_queue is not None:
                cost_loss = update_cost_function(model, batch, cost_optimizer, image_queue)
 
            if image_queue is not None:
                batch['imgs_queue'] = image_queue.get_queue_for_loss()
 
            ret = model(batch)
            
            rde_loss = ret.get('bge_loss', 0) + ret.get('tse_loss', 0)
            l2rm_noisy_loss = ret.get('l2rm_noisy_loss', 0)
            total_loss = rde_loss + l2rm_noisy_loss

            batch_size = batch['images'].shape[0]
            meters['loss'].update(total_loss.item(), batch_size)
            meters['bge_loss'].update(ret.get('bge_loss', 0), batch_size)
            meters['tse_loss'].update(ret.get('tse_loss', 0), batch_size)
            meters['cost_loss'].update(cost_loss.item() if isinstance(cost_loss, torch.Tensor) else cost_loss, batch_size)
            
            if 'l2rm_noisy_loss' not in meters:
                meters['l2rm_noisy_loss'] = AverageMeter()
            meters['l2rm_noisy_loss'].update(l2rm_noisy_loss.item() if isinstance(l2rm_noisy_loss, torch.Tensor) else l2rm_noisy_loss, batch_size)
         
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            if image_queue is not None:
                batch_label_hat = batch['label_hat']
                noisy_mask = (batch_label_hat == 0)
                if noisy_mask.sum() > 0:
                    noisy_images = batch['images'][noisy_mask]
                    image_queue.update_queue_l2rm_style(noisy_images)
                else:
                    image_queue.update_queue_after_training(batch['images'])
            
            synchronize()

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                # log loss and acc info
                for k, v in meters.items():
                    if v.avg > 0:
                        info_str += f", {k}: {v.avg:.4f}"
                        
                if 'l2rm_noisy_loss' in meters and meters['l2rm_noisy_loss'].avg > 0:
                    info_str += f" [L2RM enabled]"
                elif args.use_l2rm_noisy_loss:
                    info_str += f" [L2RM disabled - no noise data]"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)
        
 
        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        tb_writer.add_scalar('temperature', ret['temperature'], epoch)
        for k, v in meters.items():
            if v.avg > 0:
                tb_writer.add_scalar(k, v.avg, epoch)

        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))
        if epoch % eval_period == 0:
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                if args.distributed:
                    top1 = evaluator.eval(model.module.eval())
                else:
                    top1 = evaluator.eval(model.eval())

                torch.cuda.empty_cache()
                if best_top1 < top1:
                    best_top1 = top1
                    arguments["epoch"] = epoch
                    checkpointer.save("best", **arguments)
 
    if get_rank() == 0:
        logger.info(f"best R1: {best_top1} at epoch {arguments['epoch']}")

    arguments["epoch"] = epoch
    checkpointer.save("last", **arguments)
                    
def do_inference(model, test_img_loader, test_txt_loader):

    logger = logging.getLogger("RDE.test")
    logger.info("Enter inferencing")

    evaluator = Evaluator(test_img_loader, test_txt_loader)
    top1 = evaluator.eval(model.eval())
