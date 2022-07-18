import os
import logging
logger = logging.getLogger(__name__.replace('_', ''))
from scripts.utils import check_empty_gpu
if 'OMP_NUM_THREADS' not in os.environ.keys():
    max_thread = 1
    os.environ['OMP_NUM_THREADS'] = str(max_thread) # 该代码最多可建立多少个进程， 要放在最前边才能生效。防止占用过多CPU资源
    logger.warning(f' 该程序最多可建立{max_thread}个线程。')

if 'CUDA_VISIBLE_DEVICES' not in os.environ.keys(): 
    gpu_number = check_empty_gpu()
    logger.warning(f' 未指定使用的GPU，如果存在0卡，则将使用 {gpu_number} 卡。不存在就使用CPU模式')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_number)
else:
    logger.warning(f' CUDA_VISIBLE_DEVICES={os.environ["CUDA_VISIBLE_DEVICES"]}，将使用 {os.environ["CUDA_VISIBLE_DEVICES"]} 卡。')

def run_init():
    pass


