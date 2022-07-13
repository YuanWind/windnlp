import sys
from scripts.utils import set_seed
sys.path.extend(['../../','./','../'])
from scripts.main import run_init # 寻找可用的GPU
import logging
logger = logging.getLogger(__name__.replace('_', ''))
from scripts.evaluater import Evaluater, compute_objective
from scripts.trainer import MyTrainer
from scripts.config import MyConfigs, set_configs

def build_model():
    pass

def build_data():
    pass

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
 
def train_main(config_path, args=None, extra_args=None):   # sourcery skip: extract-duplicate-method
    config = set_configs(config_path, args, extra_args)
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
    train_main('projects/0_example_project/example.cfg')