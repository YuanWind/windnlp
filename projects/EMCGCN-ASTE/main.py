#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2022/07/12 19:41:01
@Author  :   Yuan Wind
@Desc    :   None
'''
import sys
sys.path.extend(['../../','./','../'])
from scripts.main import run_init # 寻找可用的GPU
import logging
logger = logging.getLogger(__name__.replace('_', ''))

from scripts.evaluater import Evaluater, compute_objective
from scripts.trainer import MyTrainer
from transformers import AutoTokenizer, set_seed
from scripts.config import MyConfigs, set_configs
from scripts.utils import load_pkl
from modules.data.datasets import ASTEDataset
from modules.data.collators import Datacollator
from modules.nn.emcgcn import EMCGCN

def build_model(kwargs=None):
    vocab = Evaluater.vocab
    config = Evaluater.config
    if kwargs is not None:
        if type(kwargs) != dict:
            for k, v in kwargs.params.items():
                config.set(k,v)
        else:
            for k, v in kwargs.items():
                config.set(k,v)
    
    post_size, deprel_size, postag_size, synpost_size, class_num = \
        vocab.num_post, vocab.num_deprel, vocab.num_postag, vocab.num_syn_post,  vocab.num_labels
    model = EMCGCN(config, post_size, deprel_size, postag_size, synpost_size, class_num)
    return model

def build_data(config, test_insts = None, train_insts=None, dev_insts=None):
    
    vocab = load_pkl(config.vocab_file)
    if test_insts is not None:
        logger.info('Change to new test insts')
        vocab.test_insts = test_insts
        
    if train_insts is not None:
        logger.info('Change to new train insts')
        vocab.train_insts = train_insts
    if dev_insts is not None:
        logger.info('Change to new dev insts')
        vocab.dev_insts = dev_insts    
        
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
    train_set = ASTEDataset(config, vocab.train_insts, tokenizer, vocab, data_type = 'train', convert_here=not config.convert_features_in_run_time)
    dev_set = ASTEDataset(config, vocab.dev_insts, tokenizer, vocab, data_type = 'dev',  convert_here=not config.convert_features_in_run_time)
    test_set = ASTEDataset(config, vocab.test_insts, tokenizer, vocab, data_type = 'test',  convert_here=not config.convert_features_in_run_time)
    data_collator = Datacollator(config, vocab, tokenizer, ASTEDataset.convert_to_features, convert_here=config.convert_features_in_run_time)
    Evaluater.vocab = vocab
    Evaluater.tokenizer = tokenizer
    return vocab,train_set,dev_set,test_set,data_collator

def hp_space(trial):
    search_space = {}
    for k in MyConfigs.hp_search_names:
        if k == "learning_rate":
            search_space[k] = trial.suggest_float("learning_rate", 1e-5, 6e-5, step=1e-5)
        elif k == "num_train_epochs":    
            search_space[k] = trial.suggest_int("num_train_epochs", 6, 20, step=2)
        elif k == "seed": 
            search_space[k] = trial.suggest_int("seed", 10, 50, step=1)
        elif k == "per_device_train_batch_size": 
            search_space[k] = trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32, 64])
        elif k == 'fgm_e':
            search_space[k] = trial.suggest_float("fgm_e", 0.1, 0.8, step=0.1)
        elif k == 'pgd_e':
            search_space[k] = trial.suggest_float("pgd_e", 0.1, 1, step=0.1)
        elif k == 'pgd_a':
            search_space[k] = trial.suggest_float("pgd_a", 0.1, 1, step=0.1)
        elif k == 'pgd_k':
            search_space[k] = trial.suggest_int("pgd_k", 1, 3, step=1)
            
    logger.info(f'搜索的超参为：{MyConfigs.hp_search_names}')
    return search_space      
 
def train_main():   # sourcery skip: extract-duplicate-method
    config = set_configs('projects/EMCGCN-ASTE/main.cfg')
    set_seed(config.seed)
    Evaluater.config = config
    vocab, train_set, dev_set, _, data_collator = build_data(config)
    Evaluater.vocab = vocab
    trainer = MyTrainer(config, 
                        model_init = build_model,
                        train_dataset=train_set, 
                        eval_dataset=dev_set, 
                        data_collator = data_collator,
                        compute_metrics=Evaluater.evaluate,
                        )

    best_run = None
    if config.do_hp_search and len(config.hp_search_name)>0: # 执行超参搜索，n_trials 是指搜索多少次
        logger.info(f'hp_search_name: {config.hp_search_name}')
        logger.info('Start hyperparameters search...\n\n')
        best_run = trainer.hyperparameter_search(hp_space=hp_space,
                                                 n_trials=config.max_trials,
                                                 compute_objective=compute_objective,
                                                 direction="maximize")
        logger.info(f'\n\nFinished hyperparameters search, the best hyperparameters: {best_run} \n\n')
        exit()
        
    if config.trainer_args.do_train and train_set is not None:
        logger.info('Reinit Model and start training...\n\n')
        trainer.train(config.resume_from_checkpoint, trial=best_run.hyperparameters if best_run is not None else None)
        logger.info(f'Finished training, save the last states to {config.best_model_file}\n\n')
    
    logger.info('---------------------------Train  Finish!  ----------------------------------\n\n')
    return config

if __name__ == '__main__': 
    train_main()