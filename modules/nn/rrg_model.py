#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@File    :   rrg_model.py
@Time    :   2022/07/21 18:57:09
@Author  :   Yuan Wind
@Desc    :   None
'''
from typing import OrderedDict

from modules.models.adalabel.modeling_adalabel import AdaLabLoss, AdaLabel_BartForConditionalGeneration, AdaLabelForConditionalGeneration, PHV_BartForConditionalGeneration
from modules.models.adalabel.configuration_adalabel import AdaLabelConfig
from transformers.models.bart import BartConfig,BartForConditionalGeneration

import torch.nn.functional as F
import torch
import torch.nn as nn
import logging
from modules.nn.base_model import BaseModel, get_parameter_names

logger = logging.getLogger(__name__.replace('_', ''))

MODEL_MAPPING = {
    'ori_ada':(AdaLabelConfig, AdaLabelForConditionalGeneration),
    'ori_bart':(BartConfig,BartForConditionalGeneration),
    'ada_bart':(BartConfig,AdaLabel_BartForConditionalGeneration),
    'ph_bart':(BartConfig,PHV_BartForConditionalGeneration),
    
}
class RRGModel(BaseModel):
    def __init__(self, config, evaluater):
        super().__init__(config)
        self.evaluater = evaluater
        # 模型
        if config.model_type == 'ori_ada':
            self.adalabel_config = AdaLabelConfig(vocab_size = config.vocab_size, dropout = config.dropout, attention_dropout=config.attn_dropout, 
                                                  pad_token_id=config.pad_token_id, bos_token_id=config.bos_token_id,
                                                  eos_token_id=config.eos_token_id, decoder_start_token_id=config.eos_token_id)
            self.bart = AdaLabelForConditionalGeneration(self.adalabel_config)
        else:
            self.adalabel_config = MODEL_MAPPING[config.model_type][0].from_pretrained(config.pretrained_model_name_or_path,
                                                                                       vocab_size = config.vocab_size,
                                                                            dropout = config.dropout, attention_dropout=config.attn_dropout,    
                                                                            pad_token_id=config.pad_token_id, bos_token_id=config.bos_token_id,
                                                                            eos_token_id=config.eos_token_id, decoder_start_token_id=config.eos_token_id)
            self.bart = MODEL_MAPPING[config.model_type][1].from_pretrained(config.pretrained_model_name_or_path, 
                                                                     config = self.adalabel_config)
        
        
        self.criterion = AdaLabLoss(self.adalabel_config.vocab_size, config.per_device_train_batch_size,
                                    ignore_index=config.pad_token_id, temperature=config.ada_temp, eos_index=config.eos_token_id)
    

    def optimizer_grouped_parameters(self, weight_decay):
        """为模型参数设置不同的优化器超参，比如分层学习率等，此处默认为hf的初始化方式
        """
        decay_parameters = get_parameter_names(self, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        another_lr_parameters = [name for name in decay_parameters if "bart" not in name]
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
    
    def forward(self, **kwargs):
        src_input_ids, tgt_input_ids = kwargs.pop('src_input_ids'), kwargs.pop('tgt_input_ids', None)
        src_attention_mask, decoder_attention_mask = kwargs.pop('src_attention_mask'), kwargs.pop('tgt_attention_mask'),
        bat_triple_idxs = kwargs.pop('bat_triple_idxs',None)
        
        decoder_input_ids = self.bart.prepare_decoder_input_ids_from_labels(tgt_input_ids)
        if self.config.model_type not in ['ph_bart']:
            outs = self.bart(src_input_ids, src_attention_mask,decoder_input_ids=decoder_input_ids, decoder_attention_mask= decoder_attention_mask, labels= None)
            
        logits, logits2 = outs[0], outs[1]
        
        # 计算损失值
        loss,loss2,nll_loss,ada_loss = self.calc_loss(logits, logits2, tgt_input_ids)
        
        generate_ids = None
        if not self.training and self.config.generate_in_eval:
            generate_ids = self.generate(src_input_ids)
        num_words = self.record_state(logits, tgt_input_ids, loss if ada_loss is None else ada_loss, logits2, loss2, nll_loss, generate_ids)
        
        forward_outs = {
            'loss': loss / num_words, 
            'logits': logits
        }
        if generate_ids is not None:
            forward_outs['generate_ids'] = generate_ids
        return forward_outs

    def generate(self, src_input_ids):
        generate_ids = self.bart.generate(src_input_ids, max_length=self.config.max_gen_length, 
                                          num_beams=self.config.num_beams,
                                          do_sample=self.config.do_sample)
        
        return generate_ids

    def calc_loss(self, generate_no_log, generate2, decoder_input_ids=None):
        if decoder_input_ids is None:
            return None
        if self.config.model_type == 'ori_bart':
            generate1 = F.log_softmax(generate_no_log, dim=-1)
            loss = F.nll_loss(generate1.reshape(-1, self.adalabel_config.vocab_size), decoder_input_ids.reshape(-1),
                                  ignore_index=self.criterion.ignore_index, reduction="sum")
            loss2 = 0
            nll_loss = loss
            ada_loss = None
        else:
            loss2 = F.cross_entropy(generate2.reshape(-1, self.adalabel_config.vocab_size), decoder_input_ids.reshape(-1),
                                    ignore_index=self.criterion.ignore_index, reduction='sum')
            
            generate1 = F.log_softmax(generate_no_log, dim=-1)
            ada_loss = self.criterion(generate1.reshape(-1, self.adalabel_config.vocab_size), decoder_input_ids.reshape(-1),
                                    decoder_input_ids, generate2.reshape(-1, self.adalabel_config.vocab_size))
            nll_loss = F.nll_loss(generate1.reshape(-1, self.adalabel_config.vocab_size), decoder_input_ids.reshape(-1),
                                    ignore_index=self.criterion.ignore_index, reduction="sum")
            loss = loss2 + ada_loss
        
        return loss,loss2,nll_loss,ada_loss
    
    def record_state(self, logits, loss_tgt_input_ids, loss, logits2, loss2, nll_loss, generate_ids):

        pred_ids = torch.argmax(logits, dim=-1)
        padding_ids = self.criterion.ignore_index
        no_padding = loss_tgt_input_ids.ne(padding_ids)
        num_correct = pred_ids.eq(loss_tgt_input_ids).masked_select(no_padding)
        correct = num_correct.sum().item()
        num = loss_tgt_input_ids.ne(self.criterion.ignore_index).sum()
        acc = correct/num
        ppl = torch.exp(loss/num)
        nll_loss = nll_loss/num
        
        label_acc, label_ppl = torch.tensor(0), torch.tensor(0)
        if self.config.model_type != 'ori_bart':
            pred1_ids = torch.argmax(logits2, dim=-1)
            num_correct2 = pred1_ids.eq(loss_tgt_input_ids).masked_select(no_padding)
            correct2 = num_correct2.sum().item()
            label_acc = correct2/num
            label_ppl = torch.exp(loss2/num)
        
        generate_acc = torch.tensor(-1)
        if generate_ids is not None:
            tmp_gene = generate_ids[:, 1:]
            try:
                tmp = torch.zeros_like(loss_tgt_input_ids)
                sp = tmp_gene.shape
                tmp[:sp[0], :sp[1]] = tmp_gene
                num_correct3 = tmp.eq(loss_tgt_input_ids).masked_select(no_padding)
                correct3 = num_correct3.sum().item()
                generate_acc = correct3/num
            except:
                tmp = torch.zeros_like(tmp_gene)
                sp = loss_tgt_input_ids.shape
                tmp[:sp[0], :sp[1]] = loss_tgt_input_ids
                no_padding = tmp.ne(padding_ids)
                num_correct3 = tmp_gene.eq(tmp).masked_select(no_padding)
                correct3 = num_correct3.sum().item()
                generate_acc = correct3/num
        
        batch_state = []
        state=OrderedDict({
            'loss': loss.item(),
            'generate_acc': generate_acc.item(),
            'gen1_acc': acc.item(),
            'nll_loss': nll_loss.item(),
            'ppl': ppl.item(),
            'gen2_acc': label_acc.item(),
            'gen2_ppl':label_ppl.item(),
            'avg_loss':(loss/num).item(),
        })
        for k ,v in state.items():
            batch_state.append(v)
        
        if not self.training:
            self.evaluater.total_loss+=loss
            self.evaluater.total_num_words+=num
            self.evaluater.total_correct+=correct
            
        self.cur_batch_state = state
        return num



