#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@File    :   rrg_model.py
@Time    :   2022/07/21 18:57:09
@Author  :   Yuan Wind
@Desc    :   None
'''
import logging
logger = logging.getLogger(__name__.replace('_', ''))
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bart import BartConfig, BartModel
from transformers.models.encoder_decoder import EncoderDecoderModel
from modules.models.adalabel.configuration_adalabel import AdaLabelConfig

from modules.models.adalabel.modeling_adalabel import AdaLabLoss, AdaLabelForConditionalGeneration


class RRGModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.adalabel_config = AdaLabelConfig(config.vocab_size,pad_token_id=0, bos_token_id=101,eos_token_id=102,
                                              decoder_start_token_id=101, forced_eos_token_id=102)
        self.bart = AdaLabelForConditionalGeneration(self.adalabel_config)
        self.criterion = AdaLabLoss(self.adalabel_config.vocab_size, 16, ignore_index=0,temperature=config.ada_temp,eos_index=102)
        
        
    def forward(self, **kwargs):
        src_input_ids, tgt_input_ids = kwargs.pop('src_input_ids'), kwargs.pop('tgt_input_ids')
        src_attention_mask, tgt_attention_mask = kwargs.pop('src_attention_mask'), kwargs.pop('tgt_attention_mask')
        
        decoder_input_ids, decoder_attention_mask = tgt_input_ids[:,:-1], tgt_attention_mask[:,:-1]
        outs = self.bart(src_input_ids, src_attention_mask,
                                         decoder_input_ids = decoder_input_ids, decoder_attention_mask = decoder_attention_mask)
        logits, logits1_log_softmax, logits2 =outs.logits, outs.generate1_log_softmax, outs.generate2
        
        generate_ids = None
        if not self.training:
            generate_ids = self.generate(src_input_ids)
            
        loss_tgt_input_ids = tgt_input_ids[:,1:]
        loss = self.calc_loss(logits1_log_softmax, logits2, loss_tgt_input_ids)
        
        pred_ids = torch.argmax(logits, dim=-1)
        padding_ids = 0
        no_padding = loss_tgt_input_ids.ne(padding_ids)
        num_correct = pred_ids.eq(loss_tgt_input_ids).masked_select(no_padding)
        correct = num_correct.sum().item()
        num = loss_tgt_input_ids.ne(0).sum()
        acc = correct/num
        batch_stat = torch.tensor([loss.item(), acc.item()])
        # logging.info(batch_stat)
        return {'loss': loss, 'logits':logits, 'batch_stat': batch_stat, 'generate_ids': generate_ids}
    
    def generate(self, src_input_ids):
        generate_ids = self.bart.generate(src_input_ids, max_length=self.config.max_gen_length,num_beams=self.config.num_beams,
                                          do_sample=self.config.do_sample)
        return generate_ids
    
    def calc_loss(self, generate1, generate2, decoder_input_ids = None):
        if decoder_input_ids is None:
            return None
        loss2 = F.cross_entropy(generate2.reshape(-1, self.adalabel_config.vocab_size), decoder_input_ids.reshape(-1), 
                                ignore_index=0)
        ada_loss = self.criterion(generate1.reshape(-1, self.adalabel_config.vocab_size), decoder_input_ids.reshape(-1), 
                                  decoder_input_ids, generate2.reshape(-1, self.adalabel_config.vocab_size))
        loss = loss2 + ada_loss
        return loss