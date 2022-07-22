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
    dump_pkl(vocab, vocab_file)
    
    


if __name__ == '__main__':
    # convert_data('projects/RRG/data/processed', 'projects/RRG/data/ori')
    # train_files = ['projects/RRG/data/processed/train.json']
    # dev_files = ['projects/RRG/data/processed/dev.json']
    # test_files = ['projects/RRG/data/processed/test.json']
    # vocab_file = 'projects/RRG/data/processed/vocab.pkl'
    # aspect2tokens = json.loads(open('projects/RRG/data/processed/aspect2tokens.json', 'r', encoding='utf-8').read())
    # build_vocab(train_files, dev_files, test_files, vocab_file, aspect2tokens)
    
    convert_data_daily('projects/RRG/data_daily')
    train_files = ['projects/RRG/data_daily/processed/train.json']
    dev_files = ['projects/RRG/data_daily/processed/dev.json']
    test_files = ['projects/RRG/data_daily/processed/test.json']
    vocab_file = 'projects/RRG/data_daily/processed/vocab.pkl'
    aspect2tokens = json.loads(open('projects/RRG/data/processed/aspect2tokens.json', 'r', encoding='utf-8').read())
    build_vocab(train_files, dev_files, test_files, vocab_file, aspect2tokens)