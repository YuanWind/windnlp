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
from transformers import BertTokenizer
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

class AdaEvaluater:
    config = None
    stage = 'dev'
    vocab = None
    tokenizer = None
    finished_idx = 0
    res = []
    total_loss = 0
    total_num_words = 0
    total_correct = 0
    @classmethod
    def evaluate(cls):
        # save result
        eval_res = cls.res
        res_file_path = cls.config.dev_pred_file if cls.stage == 'dev' else cls.config.test_pred_file
        dump_json(eval_res,res_file_path)
        
        # Metrics
        return_res = eval_generate(eval_res, cls.config)
        return_res['acc'] = (cls.total_correct/cls.total_num_words).item()
        return_res['ppl'] = torch.exp(cls.total_loss/cls.total_num_words).item()
        return_res['avg_loss'] = (cls.total_loss/cls.total_num_words).item()
        logger.info(json.dumps(return_res,indent=4))
        
        # reset 
        cls.res = []
        cls.finished_idx = 0
        cls.total_loss = 0
        cls.total_num_words = 0
        cls.total_correct = 0
        return return_res
    
    @classmethod
    def steps_evaluate(cls, preds_host, inputs_host, labels_host):
        start = cls.finished_idx
        generate_ids = preds_host[1]
        cls.finished_idx += len(generate_ids)
        end = cls.finished_idx
        insts = cls.vocab.dev_insts[start:end] if cls.stage == 'dev' else cls.vocab.test_insts[start:end]
        tokenizer = cls.tokenizer
        
        res = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)
        for idx, inst in enumerate(insts):
            inst['pred'] = res[idx]
        cls.res.extend(insts)
    
def eval_generate(insts, config):
    res_dict = {}
    golden, infer = [], []
    pred_ids, tgt_ids = [], []
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese' if config.language=='zh' else 'bert-base-uncased')
    for inst in insts:
        one_tgt_ids = tokenizer(inst['tgt'], add_special_tokens = True,
                                      padding = 'max_length', max_length = config.max_seq_len, 
                                      )
        one_pred_ids =  tokenizer(inst['pred'], add_special_tokens = True,
                                      padding = 'max_length', max_length = config.max_seq_len, 
                                      )
        pred_ids.append(one_pred_ids.input_ids)
        tgt_ids.append(one_tgt_ids.input_ids)
        
        g = tokenizer.decode(one_tgt_ids.input_ids, skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False).split(None)
        golden.append([g])
        p = tokenizer.decode(one_pred_ids.input_ids, skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False).split(None)
        infer.append(p)

    # eval bleu
    chencherry = SmoothingFunction()
    for i in range(4):
        weights = [1 / (i + 1)] * (i + 1)
        res_dict[f"BLEU-{i+1}"] = round(100 * corpus_bleu(
                golden, infer, weights=weights, smoothing_function=chencherry.method1), 2)
    
    # eval dist
    for idx, x in enumerate(calc_diversity(infer)):
        distinct = round(x * 100, 2)
        res_dict[f"dist-{idx+1}"] = distinct
    
    for idx, x in enumerate(calc_diversity([x[0] for x in golden])):
        gold_distinct = round(x * 100, 2)
        res_dict[f"ref_dist-{idx+1}"] = gold_distinct
    

    # eval ent
    ent = calc_entropy(infer)
    for idx, x in enumerate(ent):
        res_dict[f"ent-{idx+1}"] = round(x, 2)
    
    return res_dict

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

