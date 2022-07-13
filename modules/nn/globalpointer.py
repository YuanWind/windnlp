import torch
import torch.nn as nn
from torch.nn import Module, KLDivLoss
import torch.nn.functional as F
from transformers import logging

logger= logging.get_logger(__name__.replace('_', ''))


class SinusoidalPositionEmbedding(Module):
    """定义Sin-Cos位置Embedding
    """
    def __init__(
        self,
        output_dim,
        merge_mode='add',
        custom_position_ids=False
    ):
        super(SinusoidalPositionEmbedding, self).__init__()
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.custom_position_ids = custom_position_ids

    def forward(self, inputs):
        input_shape = inputs.shape
        _, seq_len = input_shape[0], input_shape[1]
        position_ids = torch.arange(seq_len).type(torch.float)[None]
        indices = torch.arange(self.output_dim // 2).type(torch.float)
        indices = torch.pow(10000.0, -2 * indices / self.output_dim)
        embeddings = torch.einsum('bn,d->bnd', position_ids, indices)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = torch.reshape(embeddings, (-1, seq_len, self.output_dim))

        if self.merge_mode == 'add':
            return inputs + embeddings.to(inputs.device)
        elif self.merge_mode == 'mul':
            return inputs * (embeddings + 1.0).to(inputs.device)
        elif self.merge_mode == 'zero':
            return embeddings.to(inputs.device)
        
class GlobalPointerCrossEntropy(nn.Module):
    '''Multi-class Focal loss implementation'''
    def __init__(self, delta = -1 , alpha = -1):
        super(GlobalPointerCrossEntropy, self).__init__()
        self.delta = delta
        self.kl_loss = KLDivLoss()
        self.alpha = alpha
        if self.alpha != -1: 
            logger.info(f'Activate rdrop! init alpha={alpha}')
        if self.delta != -1: 
            logger.info(f'Activate adpative sparse softmax! init delta={delta}')
            
    @staticmethod
    def multilabel_categorical_crossentropy(y_true, y_pred):
        
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1e12
        y_pred_pos = y_pred - (1 - y_true) * 1e12
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1) # 非标签类的得分
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1) # 标签类的得分
        
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

        return neg_loss + pos_loss

    def forward(self, logits, target):
        """
        logits: [batci_size, num_labels, seq_len, seq_len]
        """
        target = target.to_dense()
        batch_size, num_labels, seq_len,_ = logits.size()
        bh = batch_size*num_labels
        if self.delta != -1:
            logits = self.do_as(logits,target,batch_size,num_labels,seq_len)
            
        target = torch.reshape(target, (bh, -1))
        logits = torch.reshape(logits, (bh, -1))
        loss1 = torch.mean(GlobalPointerCrossEntropy.multilabel_categorical_crossentropy(target, logits))
        loss2 = None
        if self.training and self.alpha != -1: # 计算KL loss
            loss2 = self.calc_KL(logits, batch_size,num_labels)
        
        loss = loss1 + torch.mean(loss2)/4*self.alpha if loss2 is not None else loss1
        return loss
    
    def do_as(self,logits,target,batch_size,num_labels,seq_len):
        # 标签类中概率最小的 -  非标签类的概率 > delta 的logits置为 -inf, 丢掉
        delta = self.delta
        as_logits = logits.reshape(batch_size,-1)
        as_target = target.reshape(batch_size,-1)
        logits_softmax = nn.Softmax(dim=-1)(as_logits.detach())
        gold_softmax = logits_softmax  + (1-as_target) * 1e12  # 位于target=1(标签类)位置的值不变, 其他的(非标签类)值为正无穷
        not_gold_softmax = logits_softmax  + (as_target) * 1e12 # 位于target=0(非标签类)位置的值不变, 其他的(标签类)值为正无穷
        if torch.min(gold_softmax) < 1e11: # 如果没有金标，那么所有值都是1e12, 就不用做下边的步骤
            min_ = torch.min(gold_softmax,dim=-1) # 标签类中概率最小值
            not_gold_softmax = min_[0].unsqueeze(1) - not_gold_softmax # 最小值-正无穷=负无穷，就不满足>delta
            as_logits = torch.where(not_gold_softmax > delta, 
                                    torch.tensor(-1e12).type_as(as_logits), 
                                    as_logits
                                    )
            logits = as_logits.reshape(batch_size,num_labels,seq_len,seq_len)
        return logits
    
    def calc_KL(self, logits, batch_size,num_labels):
        bh = batch_size*num_labels
        logits = torch.reshape(logits, (batch_size,num_labels,-1))
        logits1,logits2 = logits[::2],logits[1::2]
        logits1 = torch.reshape(logits1, (bh//2,-1))
        logits2 = torch.reshape(logits2, (bh//2,-1))
        logits1 = F.softmax(logits1,dim=-1)+1e-12
        logits2 = F.softmax(logits2,dim=-1)+1e-12
        result = self.kl_loss(logits1.log(), logits2) + self.kl_loss(logits2.log(), logits1)
        return result


def sequence_masking(x, mask, value='-inf', axis=None):
    if mask is None:
        return x
    if value == '-inf':
        value = -65503 # 由于半精度, 最小值是65503
        # value = -1e12
    elif value == 'inf':
        value = 65503
        # value = 1e12
    assert axis > 0, 'axis must be greater than 0'
    for _ in range(axis - 1):
        mask = torch.unsqueeze(mask, 1)
    for _ in range(x.ndim - mask.ndim):
        mask = torch.unsqueeze(mask, mask.ndim)
    return x * mask + value * (1 - mask)


def prepare_as_softmax(active_logits, active_labels):   
    active_logits = active_logits.view(-1, active_logits.size(-1))        # [bat_size, num_labels]
    active_labels = active_labels.view(-1)                                # [bat_size]
    logits_softmax = nn.Softmax(dim=-1)(active_logits)        
    as_label_mask = active_labels != -100       # [10,2,3,4]
    as_active_labels = torch.where(as_label_mask, active_labels, 0 * active_labels) #        
    gold_softmax = torch.gather(logits_softmax, dim=1, index=as_active_labels.view(-1).unsqueeze(1))        
    is_lt = (gold_softmax.repeat(1, active_logits.shape[1])-logits_softmax) <= 0.25      
    as_z = torch.where(is_lt, active_logits, torch.tensor(float('-inf')).type_as(active_logits))        
    
    return as_z, active_labels


def add_mask_tril(logits, mask):
    if mask.dtype != logits.dtype:
        mask = mask.type(logits.dtype)
    logits = sequence_masking(logits, mask, '-inf', logits.ndim - 2)
    logits = sequence_masking(logits, mask, '-inf', logits.ndim - 1)
    # 排除下三角
    mask = torch.tril(torch.ones_like(logits), diagonal=-1)
    logits = logits - mask * 1e12
    return logits


class GlobalPointer(Module):
    """全局指针模块
    将序列的每个(start, end)作为整体来进行判断
    """
    def __init__(self, heads, head_size, hidden_size, RoPE=True):
        super(GlobalPointer, self).__init__()
        self.heads = heads
        self.head_size = head_size
        self.RoPE = RoPE
        self.dense = nn.Linear(hidden_size, self.head_size * self.heads * 2)


    def forward(self, inputs, mask=None):
        inputs = self.dense(inputs)
        inputs = torch.split(inputs, self.head_size * 2, dim=-1)
        # 按照-1这个维度去分，每块包含x个小块
        inputs = torch.stack(inputs, dim=-2)
        # 沿着一个新维度对输入张量序列进行连接。 序列中所有的张量都应该为相同形状
        qw, kw = inputs[..., :self.head_size], inputs[..., self.head_size:]
        # 分出qw和kw
        # RoPE编码
        if self.RoPE:
            pos = SinusoidalPositionEmbedding(self.head_size, 'zero')(inputs)
            cos_pos = pos[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos[..., None, ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], 4)
            qw2 = torch.reshape(qw2, qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], 4)
            kw2 = torch.reshape(kw2, kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
        # 计算内积
        logits = torch.einsum('bmhd , bnhd -> bhmn', qw, kw)
        # 排除padding 排除下三角
        logits = add_mask_tril(logits, mask)

        # scale返回
        return logits / self.head_size ** 0.5


class EfficientGlobalPointer(Module):
    """全局指针模块
    将序列的每个(start, end)作为整体来进行判断
    """
    def __init__(self, heads, head_size, hidden_size, RoPE=True):
        super(EfficientGlobalPointer, self).__init__()
        self.heads = heads
        self.head_size = head_size
        self.RoPE = RoPE
        self.dense_1 = nn.Linear(hidden_size, self.head_size * 2)
        self.dense_2 = nn.Linear(self.head_size * 2, self.heads * 2)

    def forward(self, inputs, mask=None):
        inputs = self.dense_1(inputs)  # batch,
        # 沿着一个新维度对输入张量序列进行连接。 序列中所有的张量都应该为相同形状
        qw, kw = inputs[..., :self.head_size], inputs[..., self.head_size:]
        # 分出qw和kw
        # RoPE编码
        if self.RoPE:
            pos = SinusoidalPositionEmbedding(self.head_size, 'zero')(inputs)
            cos_pos = pos[..., 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos[..., ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], 3)
            qw2 = torch.reshape(qw2, qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], 3)
            kw2 = torch.reshape(kw2, kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
        # 计算内积
        logits = torch.einsum('bmd , bnd -> bmn', qw, kw) / self.head_size ** 0.5
        bias = torch.einsum('bnh -> bhn', self.dense_2(inputs)) / 2
        logits = logits[:, None] + bias[:, :self.heads, None] + bias[:, self.heads:, :, None]
        # 排除padding 排除下三角
        logits = add_mask_tril(logits, mask)

        # scale返回
        return logits