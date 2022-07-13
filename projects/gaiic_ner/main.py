#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@File    :   Main.py
@Time    :   2022/05/16 12:13:40
@Author  :   Yuan Wind
@Desc    :   None
'''
import os
from collections import defaultdict
import logging
import jieba
from scripts.evaluater import compute_objective
from scripts.config import set_configs, MyConfigs
from scripts.trainer import MyTrainer
from modules.data.tokenizers import SpanTokenizer
from modules.data.datasets import NERDataset
from modules.data.collators import NERDatacollator
from scripts.evaluater import Evaluater
from modules.nn.globalpointer_ner import GPNER
from scripts.utils import convert_span_labeled_2_bio, load_json, load_pkl, set_seed

logger = logging.getLogger(__name__.replace('_', ''))

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
                
    model = GPNER(config, vocab.num_labels, vocab.cls_num, vocab.bio_num)
    return model

def build_data(config, test_insts = None, train_insts=None, dev_insts=None):
    
    vocab = load_pkl(config.vocab_file)
    if test_insts is not None:
        logger.info('Change to new test insts')
        vocab.test_insts = test_insts
    else:
        vocab.test_insts = vocab.dev_insts
        
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
        # random.shuffle(vocab.train_insts)  
        logger.info(f'train num:{len(vocab.train_insts)}, 前1w做训练集来搜索超参数')
        # vocab.dev_insts = vocab.train_insts[-10000:]
        vocab.train_insts = vocab.train_insts[:10000]
    
    if config.add_dev_data_to_train:
        vocab.train_insts = vocab.train_insts + vocab.dev_insts
        
    entity2tps = calc_entity_counts(config, vocab.train_insts)
    vocab.entity2types = entity2tps
    jieba.load_userdict(config.entity_vocab)
    
    tokenizer = SpanTokenizer(vocab=config.pretrained_model_name_or_path, max_seq_len=config.max_seq_len)
    train_set = NERDataset(config, vocab.train_insts, tokenizer, convert_here=False)
    dev_set = NERDataset(config, vocab.dev_insts, tokenizer, convert_here=False)
    test_set = NERDataset(config, vocab.test_insts, tokenizer, convert_here=False)
    data_collator = NERDatacollator(config, tokenizer, vocab, convert_here=True)
    Evaluater.vocab = vocab
    Evaluater.tokenizer = tokenizer
    return vocab,train_set,dev_set,test_set,data_collator

def pred_json2submit(pred_json_file, submit_res = None):
    pred_json = load_json(pred_json_file)
    for data in pred_json:
        data['labels'] = data['pred_labels']
        data.pop('pred_labels')
    submit_file = f'{pred_json_file}.txt' if submit_res is None else submit_res
    convert_span_labeled_2_bio(pred_json, write_file = submit_file)



def calc_entity_counts(config, datalist):
    entity2counts = defaultdict(int)
    entity2tps = defaultdict(set)
    for data in datalist:
        for lbl in data['labels']:
            tp = lbl['type']
            entity = lbl['entity']
            entity2counts[entity] += 1
            entity2tps[entity].add(tp)
    with open(config.entity_vocab, 'w', encoding='utf-8') as f:
        for k,v in entity2counts.items():
            f.write(f'{k} {v}\n')
    logger.info(f'Save jieba dictionary to {config.entity_vocab}')

    return entity2tps


 
def model_soups_main(config = None, 
                     model_params_files = None, 
                     test_file = 'data_gaiic_wind/temp_data/data_dev.json', 
                     test_pred_file = 'data_gaiic_wind/temp_data/data_dev_pred.json', 
                     batch_size = 32):
    
    if config is None:    
        config = set_configs('code/configs/offline.cfg')
    config.set('per_device_eval_batch_size',batch_size)
    config.set('eval_accumulation_steps',1)
    config.set('test_pred_file', test_pred_file)
    config.set('fp16', False)
    set_seed(config.seed)
    logger.info('Running model soups......')
    Evaluater.config = config
    Evaluater.stage = 'test'
    test_insts = load_json(test_file)
    vocab, train_set, dev_set, test_set, data_collator = build_data(config, test_insts)
    Evaluater.vocab = vocab
    trainer = MyTrainer(config, 
                        model_init = build_model,
                        train_dataset=train_set, 
                        eval_dataset=dev_set, 
                        data_collator = data_collator,
                        compute_metrics=Evaluater.evaluate,
                        )
    
    # 需要将model_params_files按照各自验证集得分降序排列
    if model_params_files is None:
        model_params_files = [
            # 'data_gaiic_wind/output_dir/output_fgm_rdrop/best_model/best.pt',
            # 'data_gaiic_wind/output_dir/output_rdrop/best_model/best.pt',
            'data_gaiic_wind/output_dir/output_pgd/best_model/best.pt',
            'data_gaiic_wind/output_dir/output_fgm/best_model/best.pt',
            'data_gaiic_wind/output_dir/output_ema/best_model/best.pt',
        ]
    
    trainer.model_soups(model_params_files, test_set)


def ensemble_main(config = None, 
                  model_params_files = None, 
                  test_file = 'data_gaiic_wind/temp_data/unlabel_data_102w.json', 
                  test_pred_file='data_gaiic_wind/temp_data/unlabel_data_102w_labels.json', 
                  batch_size = 32,
                  label_names = None):
    
    if config is None:    
        config = set_configs('code/configs/offline.cfg')
    config.set('per_device_eval_batch_size',batch_size)
    config.set('label_names',label_names)
    config.set('eval_accumulation_steps',1)
    config.set('test_pred_file', test_pred_file)
    config.set('fp16', False)
    set_seed(config.seed)
    logger.info('Running model ensemble......')
    Evaluater.config = config
    Evaluater.stage = 'test'
    test_insts = load_json(test_file)
    vocab, train_set, dev_set, test_set, data_collator = build_data(config, test_insts)
    Evaluater.vocab = vocab
    trainer = MyTrainer(config, 
                        model_init = build_model,
                        train_dataset=train_set, 
                        eval_dataset=dev_set, 
                        data_collator = data_collator,
                        compute_metrics=Evaluater.evaluate,
                        )
    
    # 需要将model_params_files按照各自验证集得分降序排列
    if model_params_files is None:
        model_params_files = [
            'data_gaiic_wind/output_dir/output_fold_0/best_model/best.pt',
            'data_gaiic_wind/output_dir/output_fold_1/best_model/best.pt',
            'data_gaiic_wind/output_dir/output_fold_2/best_model/best.pt',
            'data_gaiic_wind/output_dir/output_fold_3/best_model/best.pt',

        ]
    
    # trainer.model_soups(model_params_files, dev_set)
    trainer.ensemble_predict(model_params_files, test_set)

def hp_space(trial):
    search_space = {}
    for k in MyConfigs.hp_search_names:
        if k == "learning_rate":
            search_space[k] = trial.suggest_float("learning_rate", 1e-5, 6e-5, step=1e-5)
        elif k == "linear_lr":
            search_space[k] = trial.suggest_float("linear_lr", 2e-4, 1e-3, step=2e-4)
        elif k == "mtl_cls_w":    
            search_space[k] = trial.suggest_float("mtl_cls_w", 0, 0.8, step=0.1)
        elif k == "mtl_bio_w":    
            search_space[k] = trial.suggest_float("mtl_bio_w", 0, 0.5, step=0.1)
        elif k == "type_w":    
            search_space[k] = trial.suggest_float("type_w", 0, 0.8, step=0.1)
        elif k == "num_train_epochs":    
            search_space[k] = trial.suggest_int("num_train_epochs", 6, 20, step=2)
        elif k == "seed": 
            search_space[k] = trial.suggest_int("seed", 10, 50, step=1)
        elif k == "alpha":
            search_space[k] = trial.suggest_int("alpha", 1, 6, step=1)
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
        elif k == 'threshold':
            search_space[k] = trial.suggest_float("threshold", -1e-2, 1e-2, step=4e-3)
            
    logger.info(f'搜索的超参为：{MyConfigs.hp_search_names}')
    return search_space      
  
def train_main():   # sourcery skip: extract-duplicate-method
    config = set_configs('code/configs/debug.cfg')
    
    if os.path.exists('/home/mw/work'):
        save_path = f'/home/mw/work/best_{config.postfix}.pt'
        config.set('best_model_file', save_path)

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
        logger.info('Start training...\n\n')
        trainer.train(config.resume_from_checkpoint, trial=best_run.hyperparameters if best_run is not None else None)
        logger.info(f'Finished training, save the last states to {config.best_model_file}\n\n')
    
    logger.info('---------------------------Train  Finish!  ----------------------------------\n\n')
    return config
if __name__ == '__main__':
    config = train_main()