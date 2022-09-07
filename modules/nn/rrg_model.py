#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@File    :   rrg_model.py
@Time    :   2022/07/21 18:57:09
@Author  :   Yuan Wind
@Desc    :   None
'''
from typing import OrderedDict

from modules.models.adalabel.modeling_adalabel import  RRGBartForConditionalGeneration
from modules.models.adalabel.configuration_adalabel import AdaLabelConfig
from transformers.models.bart import BartConfig,BartForConditionalGeneration
from transformers.utils import ModelOutput

import torch.nn.functional as F
import torch
import torch.nn as nn
import logging
from modules.nn.base_model import BaseModel, get_parameter_names

logger = logging.getLogger(__name__.replace('_', ''))

# MODEL_MAPPING = {
#     'ori_bart':(BartConfig.from_pretrained,RRGBartForConditionalGeneration.from_pretrained),
# }
class RRGModel(BaseModel):
    def __init__(self, config, evaluater):
        super().__init__(config)
        self.evaluater = evaluater
        self.bart_config = BartConfig.from_pretrained(
                                    config.pretrained_model_name_or_path,vocab_size = config.vocab_size,
                                    dropout = config.dropout, attention_dropout=config.attn_dropout,    
                                    pad_token_id=config.pad_token_id, bos_token_id=config.bos_token_id,
                                    eos_token_id=config.eos_token_id, decoder_start_token_id=config.eos_token_id
                                    )
        self.bart_config.mycfg = config
        self.bart = RRGBartForConditionalGeneration.from_pretrained(config.pretrained_model_name_or_path,config = self.bart_config)
        # if config.add_ada:
        #     self.criterion = AdaLabLoss(self.bart_config.vocab_size, config.per_device_train_batch_size,
        #                             ignore_index=config.pad_token_id, temperature=config.ada_temp, eos_index=config.eos_token_id)
        self.bart.post_init()
        if not config.share_bart_params:
            # PI不共享encoder的话，使得初始化模型参数的值相同。
            self.bart.model.pi_encoder.load_state_dict(self.bart.model.encoder.state_dict())
            if hasattr(self.bart,'tgt_encoder'):
                self.bart.tgt_encoder.load_state_dict(self.bart.model.encoder.state_dict())
        
            
    def forward(self, **kwargs):
        # 取参数值
        src_input_ids, tgt_input_ids, pi_input_ids = kwargs.pop('src_input_ids'), kwargs.pop('tgt_input_ids', None), kwargs.pop('pi_input_ids', None)
        src_attention_mask, decoder_attention_mask, pi_attention_mask = kwargs.pop('src_attention_mask'), kwargs.pop('tgt_attention_mask'),kwargs.pop('pi_attention_mask')
        tri_input_ids = kwargs.pop('tri_input_ids',None)
        tri_attention_mask = kwargs.pop('tri_attention_mask',None)
        bat_triple_idxs = kwargs.pop('bat_triple_idxs',None)
        # decoder的输入右移一位
        decoder_input_ids = self.bart.prepare_decoder_input_ids_from_labels(tgt_input_ids)
        
        # 输入到模型
        outs = self.bart(src_input_ids, src_attention_mask,
                         decoder_input_ids=decoder_input_ids, 
                         decoder_attention_mask= decoder_attention_mask,
                         pi_input_ids = pi_input_ids, 
                         pi_attention_mask=pi_attention_mask, 
                         bat_triple_idxs=bat_triple_idxs,
                         tri_input_ids = tri_input_ids,
                         tri_attention_mask = tri_attention_mask
                         )
        # 取值，logits对应decoder，logits2对应ada的bi-decoder，kl_loss对应add groups是约束隐变量的loss
        logits, logits2, kl_loss = outs[0], outs[1], outs[2]
        
        # 用类成员变量保证一个batch进行自回归生成时，batch内部每个时间步的group states相同，batch之间不同，所以每个batch结束之后要置为None
        # self.bart.gru_group_states, self.bart.group_mask = None,None
        self.bart.global_z = None
            
        # 计算损失值， loss对应总损失，loss2对应logits2的损失，nll_loss对应交叉熵损失，ada_loss对应ada损失
        if tgt_input_ids is not None: # 测试阶段没有tgt_input_ids
            loss, loss2, nll_loss, ada_loss = self.calc_loss(logits, logits2, tgt_input_ids)
        
        # 训练时的eval阶段是否要进行生成，如果是，则会花很长时间进行生成。因此不推荐训练时的eval阶段进行生成，而是训练完之后将generate_in_eval置为True再生成结果。
        generate_ids = None
        if not self.training and self.config.generate_in_eval:
            generate_ids = self.bart.generate(  src_input_ids, 
                                                attention_mask=src_attention_mask,
                                                max_length=self.config.max_gen_length, 
                                                num_beams=self.config.num_beams,
                                                do_sample=self.config.do_sample,
                                                pi_input_ids=pi_input_ids,
                                                pi_attention_mask=pi_attention_mask,
                                                bat_triple_idxs = bat_triple_idxs,
                                                tri_input_ids = tri_input_ids,
                                                tri_attention_mask = tri_attention_mask
                                            )
        
        # 记录训练阶段的一些状态到tb中
        num_words = self.record_state(logits, tgt_input_ids, loss, logits2, loss2, nll_loss, generate_ids,kl_loss, ada_loss)
        
        # 求字符平均loss，并加上KL loss。
        loss = loss / num_words + kl_loss
        
        forward_outs = {
            'loss': loss, 
            'logits': logits
        }
        
        if generate_ids is not None:
            forward_outs['generate_ids'] = generate_ids

        return forward_outs

    def calc_loss(self, logits, logits2, decoder_input_ids):
        if not self.config.add_ada:
            generate1 = F.log_softmax(logits, dim=-1)
            loss = F.nll_loss(  generate1.reshape(-1, self.bart_config.vocab_size), 
                                decoder_input_ids.reshape(-1),
                                ignore_index=self.config.pad_token_id, 
                                reduction="sum"
                            )
            loss2 = None
            nll_loss = None
            ada_loss = None
            
        else:
            loss2 = F.cross_entropy(logits2.reshape(-1, self.bart_config.vocab_size), decoder_input_ids.reshape(-1),
                                    ignore_index=self.config.pad_token_id, reduction='sum')
            
            generate1 = F.log_softmax(logits, dim=-1)
            ada_loss = self.criterion(generate1.reshape(-1, self.bart_config.vocab_size), decoder_input_ids.reshape(-1),
                                    decoder_input_ids, logits2.reshape(-1, self.bart_config.vocab_size))
            nll_loss = F.nll_loss(  generate1.reshape(-1, self.bart_config.vocab_size), decoder_input_ids.reshape(-1),
                                    ignore_index=self.config.pad_token_id, reduction="sum")
            loss = loss2+ada_loss

        return loss, loss2, nll_loss, ada_loss
    
    def record_state(self, logits, loss_tgt_input_ids, loss, logits2, loss2, nll_loss, generate_ids, kl_loss, ada_loss):
        
        pred_ids = torch.argmax(logits, dim=-1)
        padding_ids = self.config.pad_token_id
        no_padding = loss_tgt_input_ids.ne(padding_ids)
        num_correct = pred_ids.eq(loss_tgt_input_ids).masked_select(no_padding)
        correct = num_correct.sum().item()
        num = loss_tgt_input_ids.ne(self.config.pad_token_id).sum()
        acc = correct/num
        ppl = torch.exp(loss/num)
        state=OrderedDict({
            'loss': (loss/num).item(), 
            'acc': acc.item(),
            'ppl': ppl.item(),
            'kl_loss':kl_loss.item()
        })
        
        # TODO if nll_loss is not None:
        
        
        logits2_acc, logits2_ppl = torch.tensor(0), torch.tensor(0)
        if self.config.add_ada:
            pred1_ids = torch.argmax(logits2, dim=-1)
            num_correct2 = pred1_ids.eq(loss_tgt_input_ids).masked_select(no_padding)
            correct2 = num_correct2.sum().item()
            logits2_acc = correct2/num
            logits2_ppl = torch.exp(loss2/num)
            state['logits2_acc'] = logits2_acc.item()
            state['logits2_ppl'] =logits2_ppl.item()
        
        gene_acc = torch.tensor(-1)
        if generate_ids is not None: # 如果有生成
            tmp_gene = generate_ids[:, 1:] # 生成的ids第一个是102，要去掉。并且生成的长度和金标的长度不一定相等，因此需要对生成的进行padding或者对金标进行padding，以便计算acc。
            try:
                tmp = torch.zeros_like(loss_tgt_input_ids)
                sp = tmp_gene.shape
                tmp[:sp[0], :sp[1]] = tmp_gene
                num_correct3 = tmp.eq(loss_tgt_input_ids).masked_select(no_padding)
                correct3 = num_correct3.sum().item()
                gene_acc = correct3/num 
            except:
                tmp = torch.zeros_like(tmp_gene)
                sp = loss_tgt_input_ids.shape
                tmp[:sp[0], :sp[1]] = loss_tgt_input_ids
                no_padding = tmp.ne(padding_ids)
                num_correct3 = tmp_gene.eq(tmp).masked_select(no_padding)
                correct3 = num_correct3.sum().item()
                gene_acc = correct3/num
            state['gene_acc'] = gene_acc.item()

        # 非训练阶段，evaluater记录总的loss，而不是bat的loss，所以这里要进行累加。
        if not self.training:
            self.evaluater.total_loss+=loss
            self.evaluater.total_num_words+=num
            self.evaluater.total_correct+=correct
            
        self.cur_batch_state = state
        return num

    def optimizer_grouped_parameters(self, weight_decay):
        """为模型参数设置不同的优化器超参，比如分层学习率等，此处默认为hf的初始化方式
        """
        total_parameters = get_parameter_names(self, [nn.LayerNorm])
        decay_parameters = [name for name in total_parameters if "bias" not in name]
        an_lr_keywords = ['post_linear1','post_linear2','prior_linear1','prior_linear2','head','group_attn','tri_encoder_attn', 
                          'cat_linear']
        another_lr_parameters = []
        for name in total_parameters:
            if 'bart' not in name:
                another_lr_parameters.append(name)
            for keyw in an_lr_keywords:
                if keyw in name:
                    another_lr_parameters.append(name)

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if n in decay_parameters and n in another_lr_parameters],
                "weight_decay": weight_decay,
                "learing_rate": self.config.non_pretrain_lr
            },
            {
                "params": [p for n, p in self.named_parameters() if n in decay_parameters and n not in another_lr_parameters],
                "weight_decay": weight_decay 
            },
            {
                "params": [p for n, p in self.named_parameters() if n not in decay_parameters and n in another_lr_parameters],
                "weight_decay": 0.0,
                "learing_rate": self.config.non_pretrain_lr
            },
            {
                "params": [p for n, p in self.named_parameters() if n not in decay_parameters and n not in another_lr_parameters],
                "weight_decay": 0.0
            }
        ]
        return optimizer_grouped_parameters
    

