#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@File    :   preprocess.py
@Time    :   2022/07/12 15:57:52
@Author  :   Yuan Wind
@Desc    :   None
'''
import sys
sys.path.extend(['../../','./','../'])
import logging
from modules.data.vocabs import RRGVocab
from scripts.utils import dump_json, dump_pkl,load_json, load_pkl
logger = logging.getLogger(__name__.replace('_', ''))
import json



def convert_data(data_dir, ori_dir):
    
    types = {'test':[], 'dev':[],'train':[]}
    for data_type in types:
        data1 =[json.loads(i) for i in open(f'{data_dir}/rrg_{data_type}.json','r',encoding='utf-8').readlines()]
        data2 = load_json(f'{data_dir}/{data_type}_source_transfered.json')
        data3 =[i.strip().replace(' ', '') for i in open(f'{ori_dir}/{"valid" if data_type == "dev" else data_type}_target.txt').readlines()]
        data4 =[i.strip().replace(' ', '') for i in open(f'{ori_dir}/{"valid" if data_type == "dev" else data_type}_source.txt').readlines()]
        data4 = {s:i for i,s in enumerate(data4)}
        
        idx = 0
        for batch in data1:
            for one1 in batch.values():
                src1 = ''.join(data2[idx]["sentence"])
                tgt_idx =  data4.get(src1,None)
                if tgt_idx is None:
                    print(f'{src1}')
                    idx += 1
                    continue
                one_data = {'src':src1, 'tgt':data3[tgt_idx], 'triples':one1}
                types[data_type].append(one_data)
                idx += 1
        dump_json(types[data_type], f'{data_dir}/{data_type}.json')
        print(f'{data_type} has {len(data2)}, and converted {len(types[data_type])}.')
        

def convert_data_daily(data_dir):
    types = {'test':[], 'dev':[],'train':[]}
    for data_type in types:
        data1 =[i.strip() for i in open(f'{data_dir}/ori/src-{"valid" if data_type == "dev" else data_type}.txt').readlines()]
        data2 =[i.strip() for i in open(f'{data_dir}/ori/tgt-{"valid" if data_type == "dev" else data_type}.txt').readlines()]
        for idx in range(len(data1)):
            src = data1[idx]
            tgt = data2[idx]
            types[data_type].append({'src':src, 'tgt':tgt, 'triples':[]})
        dump_json(types[data_type], f'{data_dir}/processed/{data_type}.json')
        print(f'{data_type} has {len(data1)}, and converted {len(types[data_type])}.')

def build_vocab(train_files, dev_files, test_files, vocab_file, aspect2tokens):
    vocab = RRGVocab()
    vocab.read_files(train_files, 'train')
    vocab.read_files(dev_files, 'dev')
    vocab.read_files(test_files, 'test')
    vocab.build(aspect2tokens)
    print(f'Train num:{len(vocab.train_insts)}, dev num:{len(vocab.dev_insts)}, test num:{len(vocab.test_insts)}')
    dump_pkl(vocab, vocab_file)
    
def align(src_path, tgt_path, json_path):
    # 对齐转换后的数据与原数据，因为转换后的数据可能丢失一些源数据。
    fixed_num = 0 
    src = open(src_path,'r', encoding='utf-8').readlines()
    tgt = open(tgt_path,'r', encoding='utf-8').readlines()
    res = load_json(json_path)
    idx0, idx1 = 0,0
    for s,t in zip(src,tgt):
        s = s.strip().replace(' ', '')
        t = t.strip().replace(' ', '')
        if idx1<len(res) and res[idx1]['src'] == s: 
            if res[idx1]['tgt'] != t:# 如果src相等但tgt不相等，则令后者等于前者，并置triples为空
                res[idx1]['tgt'] = t
                res[idx1]['triples'] = []
                fixed_num += 1
            idx1 += 1
            continue
        one = {'src':s,'tgt':t, 'triples':[]}
        res.insert(idx1, one)
        idx1 += 1
        fixed_num += 1
    assert len(res)==len(src)
    idx1 = 0
    for s,t in zip(src,tgt):
        s = s.strip().replace(' ', '')
        t = t.strip().replace(' ', '')
        if res[idx1]['src'] != s or res[idx1]['tgt'] != t:
            print(f'{idx1} error.\n{res[idx1]["src"]}\n{s}\n{res[idx1]["tgt"]}\n{t}')
        idx1 += 1
    print(f'{json_path} fixed num: {fixed_num}.')
    dump_json(res,json_path)    

def align_main():
    root_path = 'projects/data'
    src_path, tgt_path, json_path = f'{root_path}/ori/valid_source.txt',\
                                    f'{root_path}/ori/valid_target.txt',\
                                    f'{root_path}/processed/dev.json'
    align(src_path, tgt_path, json_path)
    
    src_path, tgt_path, json_path = f'{root_path}/ori/test_source.txt',\
                                    f'{root_path}/ori/test_target.txt',\
                                    f'{root_path}/processed/test.json'
    align(src_path, tgt_path, json_path)
    
    src_path, tgt_path, json_path = f'{root_path}/ori/train_source.txt',\
                                    f'{root_path}/ori/train_target.txt',\
                                    f'{root_path}/processed/train.json'
    align(src_path, tgt_path, json_path)
    
if __name__ == '__main__':
    convert_data('projects/data/processed', 'projects/data/ori')
    align_main()  # 对齐转换后的数据与原数据，因为转换后的数据可能丢失一些源数据。
    train_files = ['projects/data/processed/train.json']
    dev_files = ['projects/data/processed/dev.json']
    test_files = ['projects/data/processed/test.json']
    vocab_file = 'projects/data/processed/vocab.pkl'
    aspect2tokens = json.loads(open('projects/data/processed/aspect2tokens.json', 'r', encoding='utf-8').read())
    build_vocab(train_files, dev_files, test_files, vocab_file, aspect2tokens)
    
    # convert_data_daily('projects/data_daily')
    # train_files = ['projects/data_daily/processed/train.json']
    # dev_files = ['projects/data_daily/processed/dev.json']
    # test_files = ['projects/data_daily/processed/test.json']
    # vocab_file = 'projects/data_daily/processed/vocab.pkl'
    # aspect2tokens = json.loads(open('projects/data/processed/aspect2tokens.json', 'r', encoding='utf-8').read())
    # build_vocab(train_files, dev_files, test_files, vocab_file, aspect2tokens)