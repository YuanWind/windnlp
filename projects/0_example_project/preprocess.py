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
from modules.data.vocabs import BaseVocab, add_labels
from scripts.utils import dump_pkl
logger = logging.getLogger(__name__.replace('_', ''))





def build_vocab(train_files, dev_files, test_files, vocab_file):
    vocab = RRGVocab()
    vocab.read_files(train_files, 'train')
    vocab.read_files(dev_files, 'dev')
    vocab.read_files(test_files, 'test')
    vocab.build()
    dump_pkl(vocab, vocab_file)
    
    


if __name__ == '__main__':
    train_files = ['projects/EMCGCN-ASTE/data/sample.json']
    dev_files = ['projects/EMCGCN-ASTE/data/sample.json']
    test_files = ['projects/EMCGCN-ASTE/data/sample.json']
    vocab_file = 'projects/EMCGCN-ASTE/data/vocab.pkl'
    build_vocab(train_files, dev_files, test_files, vocab_file)