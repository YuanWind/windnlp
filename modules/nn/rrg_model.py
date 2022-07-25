#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@File    :   rrg_model.py
@Time    :   2022/07/21 18:57:09
@Author  :   Yuan Wind
@Desc    :   None
'''
from typing import OrderedDict

from transformers import BartForConditionalGeneration
from modules.models.adalabel.modeling_adalabel import AdaLabLoss, AdaLabel_BartForConditionalGeneration, AdaLabelForConditionalGeneration
from modules.models.adalabel.configuration_adalabel import AdaLabelConfig
from transformers.models.bart import BartConfig
import torch.nn.functional as F
import torch
import logging
from modules.nn.base_model import BaseModel
logger = logging.getLogger(__name__.replace('_', ''))

MODEL_TYPE = {
    'ori_ada':(AdaLabelConfig, AdaLabelForConditionalGeneration),
    'ori_bart':(BartConfig,BartForConditionalGeneration),
    'ada_bart':(BartConfig,AdaLabel_BartForConditionalGeneration)
}
class RRGModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.adalabel_config = MODEL_TYPE[config.model_type][0].from_pretrained(config.pretrained_model_name_or_path,
            vocab_size = config.vocab_size, pad_token_id=config.pad_token_id,
            bos_token_id=config.bos_token_id, eos_token_id=config.eos_token_id,
            decoder_start_token_id=config.bos_token_id, forced_eos_token_id=config.eos_token_id)
            
        self.bart = MODEL_TYPE[config.model_type][1].from_pretrained(config.pretrained_model_name_or_path, 
                                                                     config = self.adalabel_config)
        
        self.criterion = AdaLabLoss(self.adalabel_config.vocab_size, config.per_device_train_batch_size,
                                    ignore_index=config.pad_token_id, temperature=config.ada_temp, eos_index=config.eos_token_id)
        self.step = 0
        
    def forward(self, **kwargs):
        src_input_ids, tgt_input_ids = kwargs.pop('src_input_ids'), kwargs.pop('tgt_input_ids', None)
        src_attention_mask, tgt_attention_mask = kwargs.pop('src_attention_mask'), kwargs.pop('tgt_attention_mask', None)

        decoder_input_ids, decoder_attention_mask = tgt_input_ids[:,:-1], tgt_attention_mask[:, :-1]
        outs = self.bart(src_input_ids, src_attention_mask,
                         decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask)
        logits, logits2 = outs[0], outs[1]
        loss_tgt_input_ids = tgt_input_ids[:, 1:]
        loss,loss2,nll_loss = self.calc_loss(logits, logits2, loss_tgt_input_ids)
        
        generate_ids = None
        if not self.training:
            generate_ids = self.generate(src_input_ids)
        self.step += 1
        batch_stat, num_words = self.record_state(logits, loss_tgt_input_ids, loss, logits2, loss2, nll_loss, generate_ids)
        # logging.info(batch_stat)
        return {'loss': loss / num_words, 'logits': logits, 'batch_stat': batch_stat, 'generate_ids': generate_ids}

    def generate(self, src_input_ids):
        generate_ids = self.bart.generate(src_input_ids, max_length=self.config.max_gen_length, num_beams=self.config.num_beams,
                                          do_sample=self.config.do_sample)
        
        return generate_ids

    def calc_loss(self, generate_no_log, generate2, decoder_input_ids=None):
        if decoder_input_ids is None:
            return None
        loss2 = F.cross_entropy(generate2.reshape(-1, self.adalabel_config.vocab_size), decoder_input_ids.reshape(-1),
                                ignore_index=0, reduction='sum')
        
        generate1 = F.log_softmax(generate_no_log, dim=-1)
        ada_loss = self.criterion(generate1.reshape(-1, self.adalabel_config.vocab_size), decoder_input_ids.reshape(-1),
                                  decoder_input_ids, generate2.reshape(-1, self.adalabel_config.vocab_size))
        nll_loss = F.nll_loss(generate1.reshape(-1, self.adalabel_config.vocab_size), decoder_input_ids.reshape(-1),
                                  ignore_index=self.criterion.ignore_index, reduction="sum")
        loss = loss2 + ada_loss
        
        return loss,loss2,nll_loss
    
    def record_state(self, logits, loss_tgt_input_ids, loss, logits2, loss2, nll_loss, generate_ids):

        pred_ids = torch.argmax(logits, dim=-1)
        padding_ids = 0
        no_padding = loss_tgt_input_ids.ne(padding_ids)
        num_correct = pred_ids.eq(loss_tgt_input_ids).masked_select(no_padding)
        correct = num_correct.sum().item()
        num = loss_tgt_input_ids.ne(0).sum()
        acc = correct/num
        ppl = loss/num
        nll_loss = nll_loss/num
        
        
        
        pred1_ids = torch.argmax(logits2, dim=-1)
        num_correct2 = pred1_ids.eq(loss_tgt_input_ids).masked_select(no_padding)
        correct2 = num_correct2.sum().item()
        label_acc = correct2/num
        label_ppl = loss2/num
        
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
            'acc': acc.item(),
            'nll_loss': nll_loss.item(),
            'ppl': ppl.item(),
            'label_acc': label_acc.item(),
            'label_ppl':label_ppl.item(),
            'avg_loss':(loss/num).item(),
        })
        for k ,v in state.items():
            batch_state.append(v)
        
        self.cur_batch_state = state
        return torch.tensor(batch_state), num



