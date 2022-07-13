# -*- encoding: utf-8 -*-
'''
@File    :   Loss_func.py
@Time    :   2022/04/27 17:38:37
@Author  :   Yuan Wind
@Desc    :   None
'''
import logging
import torch
logger = logging.getLogger(__name__.replace('_', ''))

def GPCE(y_true, y_pred):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1) # 非标签类的得分
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1) # 标签类的得分
    
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

    return neg_loss + pos_loss


def GPCE_soft(y_true,y_pred,epsilon=1e-7,infinity = -1e12):
    y_mask = y_pred > -infinity / 10
    n_mask = (y_true < 1 - epsilon) & y_mask
    p_mask = (y_true > epsilon)& y_mask
    y_true = torch.clamp(y_true,epsilon,1 - epsilon)
    infs = torch.zeros_like(y_pred) + infinity
    y_pred_neg = torch.where(n_mask,(y_pred * 1.0),-infs) + torch.log(1 - y_true)
    y_pred_pos = torch.where(p_mask,-(y_pred * 1.0),-infs) + torch.log(y_true)
    zeros = torch. zeros_like(y_pred[..., : 1])
    y_pred_neg = torch.cat([y_pred_neg, zeros],dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg,dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    