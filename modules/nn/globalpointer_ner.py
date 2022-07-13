# -*- encoding: utf-8 -*-
'''
@File    :   GPNER.py
@Time    :   2022/04/25 18:27:59
@Author  :   Yuan Wind
@Desc    :   None
'''
import logging
import torch
from modules.nn.crf import CRF
from modules.nn.loss_func import GPCE, GPCE_soft
from torch import nn
from modules.models.gaiic_nezha.nezha_model import Gaiic_NeZhaModel
from modules.nn.globalpointer import GlobalPointer
logger = logging.getLogger(__name__.replace('_', ''))


class GPNER(nn.Module):
    def __init__(self, config, num_labels, cls_num, bio_num):
        super().__init__()
        self.config = config
        self.num_labels = num_labels
        self.cls_num = cls_num
        self.bio_num = bio_num
        logger.info(f'Use pretrain model: {config.pretrained_model_name_or_path}')
        self.bert = Gaiic_NeZhaModel.from_pretrained(config.pretrained_model_name_or_path,
                                              num_labels=self.num_labels,
                                              hidden_dropout_prob = config.hidden_dropout_prob,
                                              attention_probs_dropout_prob = config.attention_probs_dropout_prob,
                                              add_type_emb = False,
                                              type_num = self.num_labels,
                                              type_w = self.config.type_w,
                                              max_position_embeddings = config.max_position_embeddings,
                                              max_relative_position = config.max_relative_position
                                             )
        if config.add_type_emb:
            self.entity_embedding = nn.Embedding(self.num_labels, self.config.type_w, padding_idx=0)
            self.global_pointer = GlobalPointer(self.num_labels,config.head_size,config.hidden_size+15*self.config.type_w)
            nn.init.xavier_uniform_(self.global_pointer.dense.weight)
            # self.gp_linear = nn.Linear(self.bert.config.hidden_size+150, self.bert.config.hidden_size)
        # if config.add_type_emb and config.add_type_to_decoder:
        #     self.gp_linear = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        #     self.gp_linear.weight = self.bert.embeddings.entity_embedding.weight
        else:
            if self.config.use_gp:
                self.global_pointer = GlobalPointer(self.num_labels,config.head_size,config.hidden_size)
                nn.init.xavier_uniform_(self.global_pointer.dense.weight)
            else:
                self.crf_linear = nn.Linear(self.bert.config.hidden_size, self.num_labels)
                # self.bio_linear.weight = self.bert.embeddings.entity_embedding
                self.crf = CRF(num_tags=self.num_labels, batch_first=True)
                nn.init.xavier_uniform_(self.crf_linear.weight)
        
            
            
        if config.mtl_cls_w > 0:
            self.cls_linear = nn.Linear(self.bert.config.hidden_size,self.cls_num)
            nn.init.xavier_uniform_(self.cls_linear.weight)

        if config.mtl_bio_w > 0:
            self.bio_linear = nn.Linear(self.bert.config.hidden_size, self.bio_num)
            # self.bio_linear.weight = self.bert.embeddings.entity_embedding
            self.crf = CRF(num_tags=self.bio_num, batch_first=True)
            nn.init.xavier_uniform_(self.bio_linear.weight)
            
        
        
    
    def do_gp(self, kwargs, logits, labels):
        # 开始计算global pointer 
        input_ids = kwargs.get('input_ids') # 16  32
        attention_mask = kwargs.get('attention_mask')
        token_type_ids = kwargs.get('token_type_ids')
        type_ids = kwargs.get('type_ids', None)

        _,pooled_output,outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            type_ids = type_ids,
            return_dict=False,
            output_hidden_states=True
        )

        sequence_output = outputs[-1]
        
        if self.config.add_type_emb and type_ids is not None: # type_ids: [bat, seq_len, 15]  
            entity_embed = self.entity_embedding(type_ids)
            
            split_entity = torch.split(entity_embed,1,dim=2)
            split_entity = [i.squeeze(2) for i in split_entity]
            entity_embed = torch.cat(split_entity, dim=-1)
            sequence_output = torch.cat([sequence_output, entity_embed],dim=-1)
        if self.config.use_gp:
            logit1 = self.global_pointer(sequence_output, mask=attention_mask)
        else:
            logit1 = self.crf_linear(sequence_output)
            
        logits['gp_logit'] = logit1
        
        # 结束计算global pointer , 返回共享的参数
        return pooled_output,sequence_output
    
    def do_cls(self, logits, pooled_output):
        logit2 = self.cls_linear(pooled_output)
        logits['cls_logit'] = logit2
        
    def do_bio(self, logits, sequence_output):
        logit3 = self.bio_linear(sequence_output)
        logits['bio_logit'] = logit3
    
    def repeat_data(self, model_params, times):
        for k in model_params:
            if type(model_params[k][0]) is torch.Tensor:
                model_params[k] = torch.cat([model_params[k]]*times, dim=0)
            else:
                assert type(model_params[k]) == list
                model_params[k] = model_params[k] * times
                
    def forward(self, **kwargs):
        # sourcery skip: merge-dict-assign, move-assign-in-block
        model_outputs, logits, labels ={}, {}, {}
        
        if self.config.train_data_repeat > 1 and self.training:
            self.repeat_data(kwargs, self.config.train_data_repeat)
        
        pooled_output,sequence_output = self.do_gp(kwargs, logits, labels)
        
        # 类目分类
        if self.config.mtl_cls_w > 0:
            self.do_cls(logits, pooled_output)
        
        # BIO分词
        if self.config.mtl_bio_w > 0:
            self.do_bio(logits, sequence_output)
        
        # 开始计算 rdrop 的 logits
        rdrop_logits = {}
        if self.training and self.config.alpha != -1:
            pooled_output,sequence_output = self.do_gp(kwargs, rdrop_logits, labels)
            # 类目分类
            if self.config.mtl_cls_w > 0:
                self.do_cls(rdrop_logits, pooled_output)
            
            # BIO分词
            if self.config.mtl_bio_w > 0:
                self.do_bio(rdrop_logits, sequence_output)

        model_outputs['logits'] = logits['gp_logit']

        self.loss_fct(kwargs, logits, model_outputs, rdrop_logits=rdrop_logits)
        return model_outputs

    def loss_fct(self, kwargs, logits, model_outputs, rdrop_logits=None):
        labels = {'gp_label': kwargs.get('label_ids', None),
                  'cls_label': kwargs.get('cls_label', None),
                  'bio_label': kwargs.get('bio_label', None),
                  }
        loss = None
        
        # 计算主任务的损失
        if labels['gp_label'] is not None: 
            if self.config.use_gp:
                loss = self.gp_loss(logits['gp_logit'], labels['gp_label'])
            else:
                loss = self.crf(emissions = logits['gp_logit'], tags=labels['gp_label'], mask=kwargs.get('attention_mask'))
        # 计算类目分类任务的损失
        cls_loss = None
        if self.config.mtl_cls_w>0 and labels['cls_label'] is not None:
            # model_outputs['cls_logit'] = logits['cls_logit']
            # model_outputs['cls_label'] = labels['cls_label']
            active_logits = logits['cls_logit'].view(-1, self.cls_num)
            active_labels = labels['cls_label'].view(-1)
            cls_loss = nn.functional.cross_entropy(active_logits, active_labels)
        if cls_loss is not None:
            loss = loss + self.config.mtl_cls_w * cls_loss
            
        # 计算分词任务的损失
        bio_loss = None
        if self.config.mtl_bio_w>0 and labels['bio_label'] is not None:
            # model_outputs['bio_logit'] = logits['bio_logit']
            # model_outputs['bio_label'] = labels['bio_label']
            bio_loss = self.crf(emissions = logits['bio_logit'], tags=labels['bio_label'], mask=kwargs.get('attention_mask'))
        if bio_loss is not None:
            loss = loss + self.config.mtl_bio_w * bio_loss
        
        # 计算主任务 rdrop 的损失    
        if 'gp_logit' in rdrop_logits:
            rdrop_loss = self.gp_loss(rdrop_logits['gp_logit'], labels['gp_label'])
            kl_loss = self.calc_KL(logits['gp_logit'], rdrop_logits['gp_logit'])
            loss = loss + rdrop_loss + self.config.alpha*kl_loss/2
            
        if loss is not None:
            model_outputs['loss'] = loss

    
    def gp_loss(self, logits, target):
        target = target
        batch_size, num_labels, seq_len,_ = logits.size()
        bh = batch_size*num_labels
        target = torch.reshape(target, (bh, -1))
        logits = torch.reshape(logits, (bh, -1))
        gp = GPCE(target, logits)
        # gp = GPCE_soft(target, logits)
        loss = torch.mean(gp)
        return loss

    def do_as(self,logits,target,batch_size,num_labels,seq_len):
        # 标签类中概率最小的 -  非标签类的概率 > delta 的logits置为 -inf, 丢掉
        # 此处实现有点问题，待修改
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

    def calc_KL(self, logit1, logit2):
        # batch_size, num_labels = logit1.size()[0], logit1.size()[1]
        # bh = batch_size*num_labels
        # logits1,logits2 = torch.reshape(logit1, (batch_size,num_labels,-1)), torch.reshape(logit2, (batch_size,num_labels,-1))
        # logits1 = torch.reshape(logits1, (bh,-1))
        # logits2 = torch.reshape(logits2, (bh,-1))
        
        # # logits1 = torch.softmax(logits1,dim=-1)+1e-12
        # # logits2 = torch.softmax(logits2,dim=-1)+1e-12
        # logits1 = torch.sigmoid(logits1)
        # logits2 = torch.sigmoid(logits2)
        # kl_loss = torch.sum(torch.kl_div(logits1.log(), logits2),dim=-1) + torch.sum(torch.kl_div(logits2.log(), logits1),dim=-1)
        
        # kl_loss =kl_loss/2
        # kl_loss = torch.mean(kl_loss)
        
        kl_loss = gaiic_global_pointer_kl(logit1,logit2) + gaiic_global_pointer_kl(logit2,logit1)

        return kl_loss

def gaiic_global_pointer_kl(s,t):
    # ref: https://kexue.fm/archives/9039
    batch_size, num_labels = s.size()[0], s.size()[1]
    bh = batch_size*num_labels
    s = torch.reshape(s, (bh,-1))
    t = torch.reshape(t, (bh,-1))
    # s = torch.where(s)
    tmp1 = s-t
    tmp1 = torch.where(torch.isnan(tmp1), torch.full_like(tmp1, 0), tmp1) # s,t中有inf，相减之后变成了nan，这里把nan替换为0
    s1 = torch.sigmoid(s)
    t1 = torch.sigmoid(t)
    tmp2 = s1-t1
    res = tmp1*tmp2
    res = torch.sum(tmp1*tmp2,dim=-1)
    res = torch.mean(res)
    return res