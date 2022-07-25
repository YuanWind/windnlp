# -*- encoding: utf-8 -*-
'''
@File    :   Evaluater.py
@Time    :   2022/04/18 09:00:08
@Author  :   Yuan Wind
@Desc    :   None
'''
import collections
import copy
import json
import logging
import numpy as np
import torch
from scripts.utils import dump_json
logger = logging.getLogger(__name__.replace('_', ''))
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction




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
    total_loss = 0
    @staticmethod
    def evaluate():
        
        eval_res = Evaluater.res
        res_file_path = Evaluater.config.dev_pred_file if Evaluater.stage == 'dev' else Evaluater.config.test_pred_file
        dump_json(eval_res,res_file_path)
        
        return_res = eval_generate(eval_res)
        return_res['total_loss'] = Evaluater.total_loss
        logger.info(json.dumps(return_res,indent=4))
        # reset 
        Evaluater.res = []
        Evaluater.finished_idx = 0
        Evaluater.total_loss = 0
        return return_res
    
    @staticmethod
    def steps_evaluate(preds_host, inputs_host, labels_host):
        start = Evaluater.finished_idx
        Evaluater.finished_idx += len(preds_host[2])
        end = Evaluater.finished_idx
        insts = Evaluater.vocab.dev_insts[start:end] if Evaluater.stage == 'dev' else Evaluater.vocab.test_insts[start:end]
        tokenizer = Evaluater.tokenizer
        
        batch_loss,_, generate_ids = preds_host[1][0],preds_host[1][1], preds_host[2]
        Evaluater.total_loss += batch_loss.item()
        
        res = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)
        for idx, inst in enumerate(insts):
            inst['pred'] = res[idx]
        Evaluater.res.extend(insts)
    
def eval_generate(insts):
    res_dict = {}
    golden, infer = [], []
    pred_ids, tgt_ids = [], []
    for inst in insts:
        golden.append([inst['tgt'].strip().split()])
        infer.append(inst['pred'].strip().split())
        one_tgt_ids = Evaluater.tokenizer(inst['tgt'], add_special_tokens = True,
                                      padding = 'max_length', max_length = Evaluater.config.max_seq_len, 
                                      )
        one_pred_ids =  Evaluater.tokenizer(inst['pred'], add_special_tokens = True,
                                      padding = 'max_length', max_length = Evaluater.config.max_seq_len, 
                                      )
        pred_ids.append(one_pred_ids.input_ids)
        tgt_ids.append(one_tgt_ids.input_ids)
    
    ppl, acc = calc_ppl_acc(Evaluater.total_loss, pred_ids, tgt_ids)
    res_dict["ppl"] = ppl
    res_dict["acc"] = acc
    
    # eval bleu
    chencherry = SmoothingFunction()
    for i in range(4):
        weights = [1 / (i + 1)] * (i + 1)
        
        res_dict[f"BLEU-{i}"] = round(100 * corpus_bleu(
                golden, infer, weights=weights, smoothing_function=chencherry.method1), 2)
    

    # eval dist
    for idx, x in enumerate(calc_diversity(infer)):
        distinct = round(x * 100, 2)
        res_dict[f"dist-{idx}"] = distinct
    
    for idx, x in enumerate(calc_diversity([x[0] for x in golden])):
        gold_distinct = round(x * 100, 2)
        res_dict[f"ref_dist-{idx}"] = gold_distinct
    

    # eval ent
    ent = calc_entropy(infer)
    for idx, x in enumerate(ent):
        res_dict[f"ent-{idx}"] = round(x, 2)
    
    
    return res_dict

def calc_ppl_acc(loss, pred_ids, tgt_ids):
    loss = torch.tensor(loss)
    pred_ids = torch.tensor(pred_ids,dtype=int)
    tgt_ids = torch.tensor(tgt_ids,dtype=int)
    padding_ids = 0
    no_padding = tgt_ids.ne(padding_ids)
    num_correct = pred_ids.eq(tgt_ids).masked_select(no_padding)
    correct = num_correct.sum().item()
    num = tgt_ids.ne(0).sum()
    acc = correct/num
    ppl = torch.exp(min(loss/num, torch.tensor(50)))
    return ppl.item(), acc.item()
    

def calc_diversity(hyp):
    # based on Yizhe Zhang's code
    
    eps_div = 1e-10
    tokens = [0.0, 0.0]
    types = [collections.defaultdict(int), collections.defaultdict(int)]
    for line in hyp:
        for n in range(2):
            for idx in range(len(line) - n):
                ngram = ' '.join(line[idx:idx + n + 1])
                types[n][ngram] = 1
                tokens[n] += 1
    div1 = len(types[0].keys()) / (tokens[0]+eps_div)
    div2 = len(types[1].keys()) / (tokens[1]+eps_div)
    return [div1, div2]



def calc_entropy(hyps, n_lines=None):
    # based on Yizhe Zhang's code
    etp_score = [0.0, 0.0, 0.0, 0.0]
    counter = [collections.defaultdict(int), collections.defaultdict(int), collections.defaultdict(int),
               collections.defaultdict(int)]
    for line in hyps:
        for n in range(4):
            for idx in range(len(line) - n):
                ngram = ' '.join(line[idx:idx + n + 1])
                counter[n][ngram] += 1

    for n in range(4):
        total = sum(counter[n].values())
        for v in counter[n].values():
            etp_score[n] += - v / total * (np.log(v) - np.log(total))

    return etp_score













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