# -*- encoding: utf-8 -*-
'''
@File    :   Config.py
@Time    :   2022/04/18 09:01:00
@Author  :   Yuan Wind
@Desc    :   None
'''
import argparse
import logging
import platform
from typing import List
import json
from transformers import TrainingArguments
import os
from configparser import ConfigParser, ExtendedInterpolation
from dataclasses import dataclass, field
from inspect import signature
from scripts.utils import set_logger
logger = logging.getLogger(__name__.replace('_', ''))

@dataclass
class MyTrainingArguments(TrainingArguments):
    early_stop_mode: int = field(
        default=-1, metadata={"help": "-1, 0, x>0, -1: 无 early_stop 策略, 0: 连续评测4次不提升就停止， x>0: 指定训练到 >=x 轮的一次评测后停止."}
    )
    adversarival_type: str = field(
        default= None,
        metadata={"help": "[None,'fgm','pgd']."},
    )
    adv_start_steps: int = field(
        default=0, metadata={"help": "Adversarival start steps."}
    )
    fgm_e: float = field(
        default=1.0, metadata={"help": "FGM epsilon."}
    )
    pgd_e: float = field(
        default=1.0, metadata={"help": "PGD epsilon."}
    )
    pgd_a: float = field(
        default=1.0, metadata={"help": "PGD alpha."}
    )
    pgd_k: int = field(
        default=3, metadata={"help": "PGD's K."}
    )
    emb_name: str = field(
        default='emb', metadata={"help": "对那个embedding进行扰动."}
    )
    apw_e: float = field(
        default=1.0, metadata={"help": "AWP epsilon."}
    )
    awp_a: float = field(
        default=1.0, metadata={"help": "AWP alpha."}
    )
    apw_k: int = field(
        default=3, metadata={"help": "AWP 对抗步数."}
    )
    awp_param: str = field(
        default='weight', metadata={"help": "对哪些参数进行扰动."}
    )
    other_tricks: List[str] = field(
        default=None, metadata={"help": "ema, awp, swa, 'None'."}
    )
    ema_decay: float = field(
        default=0.999, metadata={"help": "ema的衰减率."}
    )
    ema_start_steps: int = field(
        default=0, metadata={"help": "ema start steps."}
    )

    swa_start: int = field(
        default=0, metadata={"help": "After swa_start optimization steps the learning rate will be switched to a constant value swa_lr."}
    )
    swa_freq: int = field(
        default=10, metadata={"help": "In the end of every swa_freq optimization steps a snapshot of the weights will be added to the SWA running average."}
    )
    swa_lr: float = field(
        default=0.001, metadata={"help": "After swa_start optimization steps the learning rate will be switched to a constant value swa_lr."}
    )
    lookahead_alpha: float = field(
        default=0.8, metadata={"help": "lookahead_alpha ."}
    )
    
    lookahead_k: int = field(
        default=3, metadata={"help": "lookahead_k."}
    )
    convert_features_in_run_time: bool = field(
        default=True, metadata={"help": "是否边训练边进行特征转换，例如tokenize等操作。False的话则一次性把全部数据转换完成再开始训练."}
    )
    
    
def set_configs(default_config_file_path = None, args=None, extra_args=None):
    """设置下方的MyConfig类，可以从config文件直接读取，也可以利用命令行传参. 优先级：命令行 > config_file
       args, extra_args 是 argparse.parse_known_args()的返回值如果外部没有使用argparser.parse_known_args(),
       则会在该方法内部设置argparse.ArgumentParser()
        
    Args:
        default_config_file_path (str, optional): 默认读取的config文件路径. Defaults to None.
        args (Namespace): args, extra_args 是 argparse.parse_known_args()的返回值.
        extra_args (list): Defaults to None.

    Returns:
        MyConfigs: 设置好参数的config
    """
    if args is None and extra_args is None:
        argsParser = argparse.ArgumentParser()
        argsParser.add_argument('--config_file', type=str, default=default_config_file_path)
        args, extra_args = argsParser.parse_known_args()
    else:
        assert type(args) == argparse.Namespace, 'args 必须为 argparse.Namespace 类型的对象'
        if not hasattr(args,'config_file'):
            args.config_file = default_config_file_path
    if args.config_file is None:
        logger.warning('没有设置config file')    
        
    if args.config_file is not None and not os.path.exists(args.config_file):
        raise FileNotFoundError(f'Config file {args.config_file} not found.')
    # 解析参数
    configs = MyConfigs(args, extra_args)

    # 设置 root logger
    set_logger(to_console=True, log_file=configs.log_file)
    logger.info(f"------------  Process ID {os.getpid()}, Process Parent ID {os.getppid()}  --------------------\n")
    configs.save()
    return configs   

class MyConfigs():
    hp_search_names = []
    def __init__(self, args=None, extra_args=None):
        # sourcery skip: hoist-if-from-if, simplify-len-comparison
        """
        初始化config，优先级为 命令行 > config file > 默认值，可以随意在config文件或者命令行里加自定义的参数。
        Args:
            config_file (str): config 文件路径.
            args (Namespace): args, extra_args 是 argparse.parse_known_args()的返回值
            extra_args (list): Defaults to None.
        """
        self.global_dir_str = None
        if args is not None:
            args = vars(args)
           
        self.config_file = args.pop('config_file')

        
        config = ConfigParser(interpolation=ExtendedInterpolation())
        if self.config_file is not None:
            config.read(self.config_file, encoding="utf-8")
            
        
            
        if extra_args:  
            # 如果命令行中有参数与config中的相同，则值使用命令行中传入的参数值
            extra_args = dict([(k[2:], v) for k, v in zip(extra_args[::2], extra_args[1::2])])
            for section in config.sections():
                for k, v in config.items(section):
                    if k in extra_args:
                        v = type(v)(extra_args.pop(k))
                        config.set(section, k, v)
                    if k in args:
                        config.set(section, k, args.pop(k))
                        
            # 如果命令行里传入了config中未定义的参数，则重新设置一个 CMD section保存这些参数
            if len(extra_args)>0:
                if 'CMD' not in config.sections():
                    config.add_section('CMD')
                for k,v in extra_args.items():
                    config.set('CMD',k,v)
                for k,v in args.items():
                    config.set(section, k, v)
         
        self._config = config
        
        sig = signature(MyTrainingArguments.__init__) # 获取 init 方法所有参数和其默认值
        self.trainer_args_dict ={k:v.default for k,v in sig.parameters.items() if k != 'self'}
        
        # 获取global_dir_str
        exist_global_dir = self.get_global_dir(config)
            
        # 按照config增加类成员变量
        for section in config.sections():
            for k, v in config.items(section):
                if self.global_dir_str in v:
                    v = v.replace(self.global_dir_str, exist_global_dir)
                
                v = self.get_type(v)
                if k in self.trainer_args_dict:
                    self.trainer_args_dict[k] = v
                self.__setattr__(k, v)

        self.post_init()
        self.trainer_args = MyTrainingArguments(**self.trainer_args_dict)
        # 如果当前是windows平台，则dataloader_num_workers需要设置为0
        if platform.system().lower() == 'windows' and self.dataloader_num_workers != 0:
            logger.warning(f'Current System is Windows and dataloader_num_workers needs to be 0 but {self.dataloader_num_workers}.')
            self.set('dataloader_num_workers', 0)
            
        
    def set(self,attr_name, attr_val):
        """
        更改或增加config的属性值
        """
        if hasattr(self.trainer_args, attr_name):
            ori_val = getattr(self.trainer_args, attr_name)
            self.trainer_args_dict[attr_name] = attr_val
            setattr(self.trainer_args, attr_name, attr_val)
            print(f'Changed config.trainer_args.{attr_name} from {ori_val} to {attr_val}.')
        if hasattr(self, attr_name):
            ori_val = getattr(self, attr_name)
            setattr(self, attr_name, attr_val)
            print(f'Changed config.{attr_name} from {ori_val} to {attr_val}')
        
        if not (hasattr(self.trainer_args, attr_name) or hasattr(self, attr_name)):
            self.__setattr__(attr_name, attr_val)
            logger.warning(f'Config has not attr {attr_name}, add config.{attr_name}: {attr_val}')
        
        
    def get_type(self, v):
        """
        设置值的类型
        """
        v = v.replace(' ', '')
        if v.lower() == 'true':
            v = True
        elif v.lower() == 'false':
            v = False
        elif v.lower() == 'none':
            v = None
        elif v == '[]':
            v = []
        elif len(v)>2 and v[0] == '[' and v[-1] == ']':
            v = v.replace('[', '')
            v = v.replace(']', '')
            v = v.split(',')
        else:
            try:
                v = eval(v)
            except Exception:
                v = v
        return v
        
    def post_init(self):
        
        MyConfigs.hp_search_names = self.hp_search_name
        if self.temp_dir is not None:
            self.temp_dir = os.path.expanduser(self.temp_dir)
            if not os.path.exists(self.temp_dir):
                os.makedirs(self.temp_dir)

        self.output_dir = self.output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        # self.best_model_dir = 
        if not os.path.exists(self.best_model_file.rsplit('/', 1)[0]):
            os.makedirs(self.best_model_file.rsplit('/', 1)[0])
            
        self.log_file = self.log_file
        if not os.path.exists(self.log_file.rsplit('/', 1)[0]):
            os.makedirs(self.log_file.rsplit('/', 1)[0])

    def get_global_dir(self, config):
        
        for section in config.sections():
            for k, v in config.items(section):
                if k == 'global_dir':
                    self.global_dir_str = v
                    break
            if self.global_dir_str is not None:
                break
        
        res = None
        if self.global_dir_str is None:
            return None
        for dir in self.get_type(self.global_dir_str):
            if os.path.exists(dir):
                res = dir
                break
        if res is None:
            raise FileNotFoundError(f'None of {self.global_dir} exists.')
        
        return res
        
                
                
    def save(self):
        logger.info(f'Loaded config file from {self.config_file} sucessfully.')
        self._config.write(open(f'{self.output_dir}/' + self.config_file.split('/')[-1], 'w'))

        logger.info(f"Write this config to {f'{self.output_dir}/' + self.config_file.split('/')[-1]} sucessfully.")

        out_str = '\n'
        for section in self._config.sections():
            for k, _ in self._config.items(section):
                out_str +='{} = {}\n'.format(k,self.__getattribute__(k))
        logger.info(out_str)
    
    def to_json_string(self):
        out_json = {}
        for section in self._config.sections():
            for k, _ in self._config.items(section):
                out_json[k] = self.__getattribute__(k)
        return json.dumps(out_json)
    
    def print(self):
        print(self.to_json_string())
        
        
if __name__ == '__main__':
    config = MyConfigs('code/configs/debug.cfg')
    config.print()
    config.set('per_device_train_batch_size', 64)
    config.print()
    config.set('per_device_train_batch_size', 8)
    config.print()