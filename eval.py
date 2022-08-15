import collections
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction,sentence_bleu
import numpy as np
from rouge import Rouge
from nltk.translate.meteor_score import meteor_score
from transformers import BertTokenizer, BartTokenizer

from scripts.utils import load_json
import torch

word2id = {'UNK':"0"}
max_id_len = 1
def eval_generate(insts,tokenizer,language='en',is_cut=False, convert2id=False):
    global max_id_len
    res_dict = {}
    golden, infer = [], []
    pred_ids, tgt_ids = [], []
    
    if language == 'zh':
        import jieba
        rouge = Rouge()
        
        for inst in insts:
            
            if not is_cut:
                gold_words = jieba.cut(inst['tgt'])
                infer_words = jieba.cut(inst['pred'])
            else:
                gold_words = inst['tgt'].split(' ')
                infer_words = inst['pred'].split(' ')
                
            if convert2id:
                gold_words_ids = []
                for w in gold_words:
                    if w not in word2id:
                        word2id[w] = str(max_id_len)
                        max_id_len += 1
                        
                    gold_words_ids.append(word2id[w])
                infer_words_ids = []
                
                for w in infer_words:
                    if w not in word2id:
                        word2id[w] = str(max_id_len)
                        max_id_len += 1
                    infer_words_ids.append(word2id.get(w, 'UNK'))
                    
                gold_words = gold_words_ids
                infer_words = infer_words_ids
                   
            golden.append(' '.join(gold_words))
            infer.append(' '.join(infer_words))
        
        # print(r_s1)
        # clothing 数据集还要计算Rouge1、Rouge2、RougeL和METEOR
        r_s = rouge.get_scores(infer, golden, avg=True)
        # print(r_s)
        res_dict[f"Rouge-1"] = r_s['rouge-1']['r'] * 100
        res_dict[f"Rouge-2"] = r_s['rouge-2']['r'] * 100
        res_dict[f"Rouge-L"] = r_s['rouge-l']['f'] * 100
        
        
        
        golden = [[i.split(' ')] for i in golden]
        infer = [i.split(' ') for i in infer]
        
        # Meteor
        ms_list = []
        for idx in range(len(infer)):
            m_s = meteor_score(golden[idx], infer[idx])
            ms_list.append(m_s)
        res_dict[f"Meteor"] = (sum(ms_list)/len(ms_list))*100
        
        # TODO 计算宏平均 BLEU
        chencherry = SmoothingFunction()
        macro_bleu_scores = []
        for idx in range(len(infer)):
            b_s = round(100 * sentence_bleu(golden[idx], infer[idx],smoothing_function=chencherry.method1), 2)
            macro_bleu_scores.append(b_s)

        res_dict[f"Macro_BLEU"] = sum(macro_bleu_scores)/len(macro_bleu_scores)
        
        bleu_score = []
        for i in range(4):
            weights = [1 / (i + 1)] * (i + 1)
            b_s = round(100 * corpus_bleu(golden, infer, weights=weights, smoothing_function=chencherry.method1), 2)
            bleu_score.append(b_s)
            res_dict[f"Micro_BLEU_{i+1}"] = b_s
        res_dict[f"Micro_BLEU"] = sum(bleu_score)/len(bleu_score)
        # eval dist
        # clothing数据只考虑前200条计算dist
        for idx, x in enumerate(calc_diversity(infer[:200])):
            distinct = round(x * 100, 2)
            res_dict[f"dist-{idx+1}"] = distinct
        
        for idx, x in enumerate(calc_diversity([x[0] for x in golden[:200]])):
            gold_distinct = round(x * 100, 2)
            res_dict[f"ref_dist-{idx+1}"] = gold_distinct
            
    else:
        for inst in insts:
            one_tgt_ids = tokenizer(inst['tgt'], add_special_tokens = True, is_split_into_words = True,
                                        padding = 'max_length', max_length = 512, 
                                        )
            one_pred_ids =  tokenizer(inst['pred'], add_special_tokens = True,
                                        padding = 'max_length', max_length = 512, 
                                        )
            pred_ids.append(one_pred_ids.input_ids)
            tgt_ids.append(one_tgt_ids.input_ids)
            g = tokenizer.decode(one_tgt_ids.input_ids, skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False).split(None)
            golden.append([g])
            
            p = tokenizer.decode(one_pred_ids.input_ids, skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False).split(None)
            infer.append(p)
        

        pred_ids = torch.tensor(pred_ids,dtype=int)
        tgt_ids = torch.tensor(tgt_ids,dtype=int)
        padding_ids = 0
        no_padding = tgt_ids.ne(padding_ids)
        num_correct = pred_ids.eq(tgt_ids).masked_select(no_padding)
        correct = num_correct.sum().item()
        num = tgt_ids.ne(0).sum()
        acc = correct/num
        print(f'acc: {acc}')
        # eval micro bleu
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
    div1 = len(types[0].keys()) / (tokens[0]+1e-12)
    div2 = len(types[1].keys()) / (tokens[1]+1e-12)
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

def test():
    tokenizer1 = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer2 = BartTokenizer.from_pretrained('facebook/bart-base')
    sent = [["well , that ' s too far . can you change some money for me ?"], ["no, it's not far from here."]]

    one_tgt_ids1 = tokenizer1(sent[0], add_special_tokens = True,
                                    padding = 'max_length', max_length = 128, 
                                    ).input_ids[0]
    one_pred_ids1 =  tokenizer1(sent[1], add_special_tokens = True,
                                    padding = 'max_length', max_length = 128, 
                                    ).input_ids[0]
    g1 = tokenizer1.decode(one_tgt_ids1, skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False).split(None)
    if ' '.join(g1)!=' '.join(sent[0]):
        print(f'\n{" ".join(g1)}\n{" ".join(sent[0])}')
    p1 = tokenizer1.decode(one_pred_ids1, skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False).split(None)
    print(p1)
    
    one_tgt_ids2 = tokenizer2(sent[0], add_special_tokens = True,
                                    padding = 'max_length', max_length = 128, 
                                    ).input_ids[0]
    one_pred_ids2 =  tokenizer2(sent[1], add_special_tokens = True,
                                    padding = 'max_length', max_length = 128, 
                                    ).input_ids[0]
    g2 = tokenizer2.decode(one_tgt_ids2, skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False).split(None)
    if ' '.join(g2)!=' '.join(sent[0]):
        print(f'\n{" ".join(g2)}\n{" ".join(sent[0])}')
    p2 = tokenizer2.decode(one_pred_ids2, skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False).split(None)
    print(p2)
if __name__=='__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    # insts = load_json('projects/RRG/outs/ada_constant/temp_dir/test_pred.json')
    # print(eval_generate(insts, tokenizer))
    # split('projects/RRG/outs/daily_100/temp_dir/test_pred_bak.json', 'projects/RRG/outs/daily_100/temp_dir')
    # test()
    path = 'projects/outs/bart_09/temp_dir/dev_pred.json'
    lang = 'zh'
    is_cut = False
    convert2id = False
    insts = load_json(path)
    res = eval_generate(insts, tokenizer, language=lang, is_cut = is_cut, convert2id = convert2id)
    res_str = f'{path} with lang={lang}, is_cut={is_cut}, convert2id={convert2id}:\n{res} \n\n'
    print(res_str)
    with open('eval_res.txt', 'a+', encoding='utf-8') as f:
        f.write(res_str)