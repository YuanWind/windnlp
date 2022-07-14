# -*- encoding: utf-8 -*-
'''
@File    :   Evaluater.py
@Time    :   2022/04/18 09:00:08
@Author  :   Yuan Wind
@Desc    :   None
'''
import copy
import json
import logging
import numpy as np
import torch
import torch.nn.functional as F
from scripts.utils import dump_json, token_index2char_index
logger = logging.getLogger(__name__.replace('_', ''))


def compute_objective(metrics):
    # 设置 optuna 超参搜索的优化目标
    metrics = copy.deepcopy(metrics)
    F1 = metrics.pop("eval_F1", None)
    if F1 is None:
        raise ValueError("F1 is None")
    return F1

class Evaluater:
    config = None
    stage = 'dev'
    vocab = None
    tokenizer = None
    finished_idx = 0
    res = []
    @staticmethod
    def evaluate():
        
        eval_res = Evaluater.res
        res_file_path = Evaluater.config.dev_pred_file if Evaluater.stage == 'dev' else Evaluater.config.test_pred_file
        dump_json(eval_res,res_file_path)
        return_res = eval_BIO(eval_res)
        logger.info(json.dumps(return_res,indent=4))
        # reset 
        Evaluater.res = []
        Evaluater.finished_idx = 0
        return return_res
    
    @staticmethod
    def steps_evaluate(preds_host, inputs_host, labels_host):
        scores= preds_host
        start = Evaluater.finished_idx
        Evaluater.finished_idx += len(preds_host)
        end = Evaluater.finished_idx
        insts = Evaluater.vocab.dev_insts[start:end] if Evaluater.stage == 'dev' else Evaluater.vocab.test_insts[start:end]
        tokenizer = Evaluater.tokenizer        
        emcgcn_decode(scores, labels_host, insts, tokenizer)
        Evaluater.res.extend(insts)

def eval_BIO(insts):
    pred_aspect_spans,pred_opinion_spans,pred_triplets_utm = [], [], []
    true_aspect_spans,true_opinion_spans,true_triplets_utm = [], [], []
    for idx, inst in enumerate(insts):
        pred_aspect_spans.extend([f'{idx}-{sp}' for sp in inst['pred_aspect_spans']])
        pred_opinion_spans.extend([f'{idx}-{sp}' for sp in  inst['pred_opinion_spans']])
        pred_triplets_utm.extend([f'{idx}-{sp}' for sp in  inst['pred_triplets_utm']])
        true_aspect_spans.extend([f'{idx}-{sp}' for sp in  inst['true_aspect_spans']])
        true_opinion_spans.extend([f'{idx}-{sp}' for sp in  inst['true_opinion_spans']])
        true_triplets_utm.extend([f'{idx}-{sp}' for sp in  inst['true_triplets_utm']])
    
    def get_f(golden_set,predicted_set):
        correct_num = len(golden_set & predicted_set)
        precision = correct_num / len(predicted_set) if len(predicted_set) > 0 else 0
        recall = correct_num / len(golden_set) if len(golden_set) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1
    
    p_aspect, r_aspect, f_aspect = get_f(set(true_aspect_spans), set(pred_aspect_spans))
    p_opinion, r_opinion, f_opinion = get_f(set(true_opinion_spans), set(pred_opinion_spans))
    p_triplets_utm, r_triplets_utm, f_triplets_utm = get_f(set(true_triplets_utm), set(pred_triplets_utm))
    
    res = {
        'p_aspect' : p_aspect, 
        'r_aspect' : r_aspect, 
        'f_aspect' : f_aspect,
        'p_opinion' : p_opinion, 
        'r_opinion' : r_opinion, 
        'f_opinion' : f_opinion,
        'p_triplets_utm' : p_triplets_utm, 
        'r_triplets_utm' : r_triplets_utm, 
        'f_triplets_utm' : f_triplets_utm 
    }
    return res

def emcgcn_decode(scores, labels, insts, tokenizer):
    token_ranges = None
    sentences = []
    for inst in insts:
        sentences.append(''.join(inst['sentence']))
    token_features = tokenizer(sentences, return_offsets_mapping=True, add_special_tokens=True,
                                padding = 'max_length', max_length = Evaluater.config.max_seq_len, return_tensors = 'pt')
    token_ranges =  [token_index2char_index(token_features[idx].offsets) for idx in range(len(insts))]
    preds = F.softmax(torch.tensor(scores, dtype=torch.float32), dim=-1)
    preds = torch.argmax(preds, dim=3)
    
    for idx, inst in enumerate(insts):
        pred_aspect_spans,pred_opinion_spans,pred_triplets_utm = find_triplet(preds[idx], token_ranges[idx])
        true_aspect_spans,true_opinion_spans,true_triplets_utm = find_triplet(torch.tensor(labels[idx], dtype=torch.long), token_ranges[idx])
        inst['pred_aspect_spans'] = pred_aspect_spans
        inst['pred_opinion_spans'] = pred_opinion_spans
        inst['pred_triplets_utm'] = pred_triplets_utm
        inst['true_aspect_spans'] = true_aspect_spans
        inst['true_opinion_spans'] = true_opinion_spans
        inst['true_triplets_utm'] = true_triplets_utm

def find_triplet(tags, token_ranges):
    # label2id = {'N': 0, 'B-A': 1, 'I-A': 2, 'A': 3, 'B-O': 4, 'I-O': 5, 'O': 6, 'negative': 7, 'neutral': 8, 'positive': 9}
    neg_id = Evaluater.vocab.label2id.get('negative')
    neu_id = Evaluater.vocab.label2id.get('neutral')
    pos_id = Evaluater.vocab.label2id.get('positive')
    triplets_utm = []
    aspect_spans = get_aspects(tags, token_ranges, ignore_index=0)
    opinion_spans = get_opinions(tags, token_ranges, ignore_index=0)
    for al, ar in aspect_spans:
        for pl, pr in opinion_spans:
            tag_num = [0] * len(Evaluater.vocab.label2id)
            for i in range(al, ar + 1):
                for j in range(pl, pr + 1):
                    a_start = token_ranges[i][0]
                    o_start = token_ranges[j][0]
                    if al < pl:
                        tag_num[int(tags[a_start][o_start])] += 1
                    else:
                        tag_num[int(tags[o_start][a_start])] += 1

            if sum([tag_num[neg_id],tag_num[neu_id],tag_num[pos_id]]) == 0: continue
            sentiment = -1
            if tag_num[pos_id] >= tag_num[neu_id] and tag_num[pos_id] >= tag_num[neg_id]:
                sentiment = pos_id
            elif tag_num[neu_id] >= tag_num[neg_id] and tag_num[neu_id] >= tag_num[pos_id]:
                sentiment = neu_id
            elif tag_num[neg_id] >= tag_num[pos_id] and tag_num[neg_id] >= tag_num[neu_id]:
                sentiment = neg_id
            if sentiment == -1:
                print('wrong!!!!!!!!!!!!!!!!!!!!')
                exit()
            triplets_utm.append([al, ar, pl, pr, sentiment])
    
    return aspect_spans,opinion_spans,triplets_utm
    


def get_aspects(tags, token_range, ignore_index=0):
    spans = []
    start, end = -1, -1
    length = len(token_range)
    for i in range(length):
        l, r = token_range[i]
        if tags[l][l] == ignore_index:
            continue
        label = Evaluater.vocab.id2label[tags[l][l]]
        if label == 'B-A':
            if start != -1:
                spans.append([start, end])
            start, end = i, i
        elif label == 'I-A':
            end = i
        else:
            if start != -1:
                spans.append([start, end])
                start, end = -1, -1
    if start != -1:
        spans.append([start, length-1])
    
    return spans


def get_opinions(tags, token_range, ignore_index=0):
    spans = []
    start, end = -1, -1
    length = len(token_range)
    for i in range(length):
        l, r = token_range[i]
        if tags[l][l] == ignore_index:
            continue
        label = Evaluater.vocab.id2label[tags[l][l]]
        if label == 'B-O':
            if start != -1:
                spans.append([start, end])
            start, end = i, i
        elif label == 'I-O':
            end = i
        else:
            if start != -1:
                spans.append([start, end])
                start, end = -1, -1
    if start != -1:
        spans.append([start, length-1])
    
    return spans
