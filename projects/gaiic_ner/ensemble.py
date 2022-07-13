import sys
sys.path.extend(['../../','./','../', 'code'])
import logging
logger = logging.getLogger(__name__.replace('_', ''))
import os
from wind_scripts.utils import check_empty_gpu

if 'OMP_NUM_THREADS' not in os.environ.keys():
    max_thread = 1
    os.environ['OMP_NUM_THREADS'] = str(max_thread) # 该代码最多可建立多少个进程， 要放在最前边才能生效。防止占用过多CPU资源
    logger.warning(f' 该程序最多可建立{max_thread}个线程。')

if 'CUDA_VISIBLE_DEVICES' not in os.environ.keys(): 
    gpu_number = check_empty_gpu()
    logger.warning(f' 未指定使用的GPU，将使用 {gpu_number} 卡。')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_number)
else:
    logger.warning(f' CUDA_VISIBLE_DEVICES={os.environ["CUDA_VISIBLE_DEVICES"]}，将使用 {os.environ["CUDA_VISIBLE_DEVICES"]} 卡。')

from wind_scripts.Config import set_configs
from wind_scripts.Main import ensemble_main,model_soups_main

if __name__ == "__main__":

    # config = set_configs()
    model_params_files = [
        'data_gaiic_wind/output_dir/output_baseline_epoch30/best_model/best.pt',
        'data_gaiic_wind/output_dir/output_baseline_cosine/best_model/best.pt',
        'data_gaiic_wind/output_dir/output_baseline_alpha2/best_model/best.pt',
        # 'data_gaiic_wind/output_dir/output_seed666/best_model/best.pt'
    ]
    # test_file = 'data_gaiic_wind/temp_data/unlabel_data_102w.json'
    # test_pred_file = 'data_gaiic_wind/temp_data/unlabel_data_102w_labels.json'
    
    test_file = 'data_gaiic_wind/temp_data/data_dev.json'
    test_pred_file = 'data_gaiic_wind/temp_data/data_dev_pred.json'
    # ensemble_main(config = None, 
    #               model_params_files = model_params_files, 
    #               test_file = test_file, 
    #               test_pred_file = test_pred_file, 
    #               batch_size = 32,
    #               label_names = None)
    
    model_soups_main(
        config = None, 
        model_params_files = model_params_files, 
        test_file = test_file, 
        test_pred_file = test_pred_file, 
        batch_size = 32,
    )
    