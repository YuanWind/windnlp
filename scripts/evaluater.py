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
from scripts.utils import dump_json
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
        P,R,F = eval_BIO(eval_res)
        P_bio,R_bio,F_bio = eval_BIO(eval_res, tp='0')
        return_res = {'P':float(P),'R':float(R),'F1': float(F),
                      'P_bio':float(P_bio),'R_bio':float(R_bio),'F1_bio': float(F_bio)} 
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
        vocab = Evaluater.vocab
        token_mappings = get_token_mappings(insts, tokenizer)
        decoded_insts = decode_entities(insts, scores, token_mappings,vocab, threshold= Evaluater.config.threshold)
        Evaluater.res.extend(decoded_insts)
    
        
def get_token_mappings(insts, tokenizer):
    token_mappings = []
    for inst in insts:
        tokens = tokenizer.tokenize(inst['text'])[:Evaluater.config.max_seq_len-2]
        token_mapping = tokenizer.get_token_mapping(inst['text'], tokens)  # token_id -> word_idxs
        token_mappings.append(token_mapping)
    return token_mappings

def to_list(labels,prefix='', tp = None):
    res = []
    for l in labels:
        res.append(f"{prefix}_{l['start_idx']}_{l['end_idx']}_{l['type'] if tp is None else tp}_{l['entity']}")
    return res
def to_set(labels,prefix=''): 
    # sourcery skip: set-comprehension
    res = set()
    for l in labels:
        res.add(f"{prefix}_{l['start_idx']}_{l['end_idx']}_{l['type']}_{l['entity']}")
    return res

def decode_entities(insts, scores, token_mappings,vocab, threshold=0): 
    scores[:,:, [0, -1]] -= np.inf
    scores[:,:, :, [0, -1]] -= np.inf
    
    
    for idx,inst in enumerate(insts):
        text = inst['text']
        entities = []
        token_mapping = token_mappings[idx]
        for category, start, end in zip(*np.where(scores[idx] > threshold)):
            if end-1 >= len(token_mapping):
                break
            if token_mapping[start-1][0] <= token_mapping[end-1][-1]:
                entitie_ = {
                    "start_idx": token_mapping[start-1][0],
                    "end_idx": token_mapping[end-1][-1],
                    "entity": text[token_mapping[start-1][0]: token_mapping[end-1][-1]+1],
                    "type": vocab.id2label[category]
                }
                if entitie_['entity'] == '':
                    continue
                entities.append(entitie_)
        
        entities = sorted(entities, key=lambda entitie:entitie['start_idx']) # 根据 start_idx 排序
        inst['pred_labels'] = entities
        
    return insts

def eval_BIO(eval_res, tp = None):
    epsilon = 1e-12
    total_pred_entities, total_true_entities = [],[]
    
    for idx, data in enumerate(eval_res):
        
        total_true_entities.extend(to_list(data['labels'], str(idx), tp = tp))
        total_pred_entities.extend(to_list(data['pred_labels'], str(idx), tp = tp))

            
    total_pred_entities, total_true_entities = set(total_pred_entities), set(total_true_entities)

    P = len(total_true_entities&total_pred_entities)/(len(total_pred_entities)+epsilon)
    R = len(total_true_entities&total_pred_entities)/(len(total_true_entities)+epsilon)
    
    F = 2*P*R/(P+R+epsilon)
    return P,R,F