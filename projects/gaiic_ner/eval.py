# -*- encoding: utf-8 -*-
'''
@File    :   submit_result.py
@Time    :   2022/04/29 18:23:13
@Author  :   Yuan Wind
@Desc    :   None
'''
import logging
logger = logging.getLogger(__name__.replace('_', ''))
import os
if not os.path.exists('code'):
    os.mkdir('code')
    
import sys
sys.path.extend(['code/'])
from wind_scripts.utils import check_empty_gpu
if 'OMP_NUM_THREADS' not in os.environ.keys():
    max_thread = 1
    os.environ['OMP_NUM_THREADS']=str(max_thread) # 该代码最多可建立多少个进程，要放在最前边才能生效。防止占用过多CPU资源
    logger.warning(f' 该程序最多可建立{max_thread}个线程。')
if 'CUDA_VISIBLE_DEVICES' not in os.environ.keys(): 
    gpu_number = check_empty_gpu()
    logger.warning(f' 未指定使用的GPU，将使用 {gpu_number} 卡。')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_number)
else:
    logger.warning(f'CUDA_VISIBLE_DEVICES={os.environ["CUDA_VISIBLE_DEVICES"]}，将使用 {os.environ["CUDA_VISIBLE_DEVICES"]} 卡。')

import torch
def unzip_files(path):
    import zipfile
    import time
    '''
    基本格式：zipfile.ZipFile(filename[,mode[,compression[,allowZip64]]])
    mode：可选 r,w,a 代表不同的打开文件的方式；r 只读；w 重写；a 添加
    compression：指出这个 zipfile 用什么压缩方法，默认是 ZIP_STORED，另一种选择是 ZIP_DEFLATED；
    allowZip64：bool型变量，当设置为True时可以创建大于 2G 的 zip 文件，默认值 True；

    '''
    print('开始解压')
    start = time.time()
    zip_file = zipfile.ZipFile(path)
    zip_list = zip_file.namelist() # 得到压缩包里所有文件
    for f in zip_list:
        if '.zip' in f:
            continue
        zip_file.extract(f, '.') # 循环解压文件到指定目录
    zip_file.close() # 关闭文件，必须有，释放内存
    os.remove(path)
    end = time.time()
    print(f'解压完成，用时：{end-start}')
    
def pred_BIO(path_word:str, path_sample:str, batch_size):

    from wind_scripts.Evaluater import Evaluater
    from wind_scripts.Trainer import MyTrainer
    from wind_scripts.utils import convert_bio_2_span_labeled, dump_json, set_seed,load_json
    from wind_scripts.Config import set_configs

    config = set_configs('code/wind_configs/submit.cfg')
    config.trainer_args.per_device_eval_batch_size  = batch_size
    config.trainer_args.fp16 = False # 测试的时候把FP16给关了，防止无法复现
    set_seed(config.seed)
    config.save()
    Evaluater.config = config
    logger.info(f'load test data from {path_word}')
    vocab, train_set, dev_set, test_set, data_collator = build_data(config)
    evaler = MyTrainer( config, 
                        model_init = build_model,
                        train_dataset=train_set, 
                        eval_dataset=dev_set, 
                        data_collator = data_collator,
                        compute_metrics=Evaluater.evaluate,
                        )

    # if config.trainer_args.do_eval and dev_set is not None:
    #     pred_load(evaler.model, config, 'Use loaded model to evaluate dev dataset:', 'dev')
    #     evaler.evaluate(dev_set)

    if config.trainer_args.do_predict and test_set is not None:
        pred_load(evaler.model, config, 'Use loaded model to predict test dataset:', 'test')
        evaler.predict(test_set)
        # pred_json2submit(config.test_pred_file, submit_res='results.txt')


def pred_load(model, config, arg2, arg3):
    from wind_scripts.Evaluater import Evaluater
    model.load_state_dict(torch.load(config.best_model_file), strict=False)
    logger.info(f'Load model state from {config.best_model_file}')
    logger.info(arg2)
    Evaluater.stage = arg3


if __name__ == "__main__":
    # pred_BIO('/home/mw/input/train_dev7893/temp_data/temp_data/0.90_dev.json', None, batch_size = 64)
    pred_BIO(None, None, batch_size = 12)