#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@File    :   rrg_model.py
@Time    :   2022/07/21 18:57:09
@Author  :   Yuan Wind
@Desc    :   None
'''
from cProfile import label
from typing import OrderedDict
from modules.models.adalabel.modeling_adalabel import AdaLabLoss, AdaLabelForConditionalGeneration
from modules.models.adalabel.configuration_adalabel import AdaLabelConfig
from transformers.models.encoder_decoder import EncoderDecoderModel
from transformers.models.bart import BartConfig, BartModel
import torch.nn.functional as F
import torch.nn as nn
import torch
import logging
logger = logging.getLogger(__name__.replace('_', ''))


class RRGModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.adalabel_config = AdaLabelConfig(config.vocab_size, pad_token_id=config.pad_token_id,
                                              bos_token_id=config.bos_token_id, eos_token_id=config.eos_token_id,
                                              decoder_start_token_id=config.bos_token_id, forced_eos_token_id=config.eos_token_id)
        self.bart = AdaLabelForConditionalGeneration(self.adalabel_config)
        self.criterion = AdaLabLoss(self.adalabel_config.vocab_size, config.per_device_train_batch_size,
                                    ignore_index=config.pad_token_id, temperature=config.ada_temp, eos_index=config.eos_token_id)
        
    def forward(self, **kwargs):
        src_input_ids, tgt_input_ids = kwargs.pop(
            'src_input_ids'), kwargs.pop('tgt_input_ids', None)
        src_attention_mask, tgt_attention_mask = kwargs.pop(
            'src_attention_mask'), kwargs.pop('tgt_attention_mask', None)

        decoder_input_ids, decoder_attention_mask = tgt_input_ids[:,:-1], tgt_attention_mask[:, :-1]
        outs = self.bart(src_input_ids, src_attention_mask,
                         decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask)
        logits, logits1_log_softmax, logits2 = outs.logits, outs.generate1_log_softmax, outs.generate2

        generate_ids = None
        if not self.training:
            generate_ids = self.generate(src_input_ids)
        loss_tgt_input_ids = tgt_input_ids[:, 1:]
        loss,loss2,nll_loss = self.calc_loss(logits1_log_softmax, logits2, loss_tgt_input_ids)
        batch_stat = self.record_state(logits, loss_tgt_input_ids, loss, logits2, loss2, nll_loss)

        # logging.info(batch_stat)
        return {'loss': loss / logits1_log_softmax.shape[0], 'logits': logits, 'batch_stat': batch_stat, 'generate_ids': generate_ids}

    def generate(self, src_input_ids):
        generate_ids = self.bart.generate(src_input_ids, max_length=self.config.max_gen_length, num_beams=self.config.num_beams,
                                          do_sample=self.config.do_sample)
        return generate_ids

    def calc_loss(self, generate1, generate2, decoder_input_ids=None):
        if decoder_input_ids is None:
            return None
        loss2 = F.cross_entropy(generate2.reshape(-1, self.adalabel_config.vocab_size), decoder_input_ids.reshape(-1),
                                ignore_index=0, reduction='sum')
        ada_loss = self.criterion(generate1.reshape(-1, self.adalabel_config.vocab_size), decoder_input_ids.reshape(-1),
                                  decoder_input_ids, generate2.reshape(-1, self.adalabel_config.vocab_size))
        nll_loss = F.nll_loss(generate1.reshape(-1, self.adalabel_config.vocab_size), decoder_input_ids.reshape(-1),
                                  ignore_index=self.criterion.ignore_index, reduction="sum")
        loss = loss2 + ada_loss
        
        return loss,loss2,nll_loss
    
    def record_state(self, logits, loss_tgt_input_ids, loss, logits2, loss2, nll_loss):
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
        batch_state = []
        state={
            'loss': loss.item(),
            'acc': acc.item(),
            'nll_loss': nll_loss.item(),
            'ppl': ppl.item(),
            'label_acc': label_acc.item(),
            'label_ppl':label_ppl.item()
        }
        for k ,v in state.items():
            batch_state.append(v)
        logger.info(state)
        return torch.tensor(batch_state)
