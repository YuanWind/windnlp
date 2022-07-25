import collections
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import numpy as np
from transformers import BertTokenizer

from scripts.utils import load_json
import torch


def eval_generate(insts,tokenizer):
    res_dict = {}
    golden, infer = [], []
    pred_ids, tgt_ids = [], []
    for inst in insts:
        golden.append([inst['tgt'].strip().split()])
        infer.append(inst['pred'].strip().split())
        one_tgt_ids = tokenizer(inst['tgt'], add_special_tokens = True,
                                      padding = 'max_length', max_length = 128, 
                                      )
        one_pred_ids =  tokenizer(inst['pred'], add_special_tokens = True,
                                      padding = 'max_length', max_length = 128, 
                                      )
        pred_ids.append(one_pred_ids.input_ids)
        tgt_ids.append(one_tgt_ids.input_ids)
    

    pred_ids = torch.tensor(pred_ids,dtype=int)
    tgt_ids = torch.tensor(tgt_ids,dtype=int)
    padding_ids = 0
    no_padding = tgt_ids.ne(padding_ids)
    num_correct = pred_ids.eq(tgt_ids).masked_select(no_padding)
    correct = num_correct.sum().item()
    num = tgt_ids.ne(0).sum()
    acc = correct/num
    print(f'acc: {acc}')
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

def calc_diversity(hyp):
    # based on Yizhe Zhang's code
    
    tokens = [0.0, 0.0]
    types = [collections.defaultdict(int), collections.defaultdict(int)]
    for line in hyp:
        for n in range(2):
            for idx in range(len(line) - n):
                ngram = ' '.join(line[idx:idx + n + 1])
                types[n][ngram] = 1
                tokens[n] += 1
    div1 = len(types[0].keys()) / (tokens[0]+1e-6)
    div2 = len(types[1].keys()) / (tokens[1]+1e-6)
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

def split(json_file, save_path):
    data = load_json(json_file)
    with open(save_path+'/valid_tgt.txt', 'w' ,encoding='utf-8') as f:
        for d in data:
            f.write(d['tgt']+'\n')
    with open(save_path+'/valid_pred.txt', 'w' ,encoding='utf-8') as f:
        for d in data:
            f.write(d['pred']+'\n')
            
if __name__=='__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    insts = load_json('projects/RRG/outs/daily_100/temp_dir/test_pred_bak.json')
    print(eval_generate(insts, tokenizer))
    # split('projects/RRG/outs/daily_100/temp_dir/test_pred_bak.json', 'projects/RRG/outs/daily_100/temp_dir')