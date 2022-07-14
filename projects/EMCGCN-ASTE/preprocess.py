#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@File    :   preprocess.py
@Time    :   2022/07/12 15:57:52
@Author  :   Yuan Wind
@Desc    :   None
'''
import os
import sys
sys.path.extend(['../../','./','../'])

from modules.data.datasets import ASTEDataset

from transformers import AutoTokenizer

import logging
from modules.data.vocabs import ASTEVocab
from scripts.utils import dump_pkl, load_pkl, set_seed
logger = logging.getLogger(__name__.replace('_', ''))
from scripts.config import set_configs

def build_vocab(train_files, dev_files, test_files, vocab_file):
    vocab = ASTEVocab()
    vocab.read_files(train_files, 'train')
    vocab.read_files(dev_files, 'dev')
    vocab.read_files(test_files, 'test')
    vocab.build()
    dump_pkl(vocab, vocab_file)
    print(f'Save vocab to {vocab_file}')
    
    
def build_dataset():
    config = set_configs('projects/EMCGCN-ASTE/main.cfg')
    set_seed(config.seed)
    vocab = load_pkl(config.vocab_file)   
        
    if config.max_train_num > 0:
        vocab.train_insts = vocab.train_insts[:config.max_train_num]
    if config.max_dev_num > 0:
        vocab.dev_insts = vocab.dev_insts[:config.max_dev_num]
    
    if config.do_hp_search:
        logger.info(f'train num:{len(vocab.train_insts)}, 前1w做训练集来搜索超参数')
        vocab.train_insts = vocab.train_insts[:10000]
    
    if config.add_dev_data_to_train:
        vocab.train_insts = vocab.train_insts + vocab.dev_insts
        
    
    tokenizer = AutoTokenizer.from_pretrained(config.pretrain_model_path)
    dataset_path = os.path.join(config.temp_dir, 'dataset_all.pkl')
    if not os.path.exists(dataset_path) or not config.load_dataset_from_pkl:
        train_set = ASTEDataset(config, vocab.train_insts, tokenizer, vocab, data_type = 'train', convert_here=not config.convert_features_in_run_time)
        dev_set = ASTEDataset(config, vocab.dev_insts, tokenizer, vocab, data_type = 'dev',  convert_here=not config.convert_features_in_run_time)
        test_set = ASTEDataset(config, vocab.test_insts, tokenizer, vocab, data_type = 'test',  convert_here=not config.convert_features_in_run_time)
        if config.save_dataset:
            dump_pkl((train_set,dev_set, test_set), dataset_path)



if __name__ == '__main__':
    # train_files = ['projects/EMCGCN-ASTE/data/train.json']
    # dev_files = ['projects/EMCGCN-ASTE/data/dev.json']
    # test_files = ['projects/EMCGCN-ASTE/data/test.json']
    # vocab_file = 'projects/EMCGCN-ASTE/data/vocab.pkl'
    # build_vocab(train_files, dev_files, test_files, vocab_file)
    build_dataset()