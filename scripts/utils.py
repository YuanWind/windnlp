# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2022/04/18 08:59:24
@Author  :   Yuan Wind
@Desc    :   None
'''
import contextlib
import logging
import os
import random
import pickle
import json
logger = logging.getLogger(__name__.replace('_', ''))


def sort_dict(d, mode='k', reverse=False): 
    """对字典按照key或者value排序

    Args:
        d (dict): 待排序的字典对象
        mode (str, optional): 'k'-->键排序, 'v'-->值排序 . Defaults to 'k'.
        reverse (bool, optional): True为降序排列. Defaults to False.

    Returns:
        list(touple): 返回一个list, 里边touple第一个为key, 第二个为value
    """
    # assert type(d) == dict, 'sort_dict仅支持对dict排序, 当前对象为:{}'.format(type(d))
    if mode == 'k': 
        return [(i, d[i]) for i in sorted(d, reverse=reverse)]
    elif mode == 'v': 
        return sorted(d.items(), key = lambda kv: kv[1], reverse=reverse)
    else:
        logger.info('排序失败')
        return d

def check_empty_gpu(find_ours=11.5, threshold_mem=5000*1000000):
    """检查GPU的空闲状态，设定判断阈值
    Args:
        find_ours (float, optional): _description_. Defaults to 11.5.
        100*1000000 = 100 M
    Returns:
        _type_: _description_
    """
    try:
        import pynvml
        import time
        start = time.time()
        find_times = 0
        pynvml.nvmlInit()
        cnt = pynvml.nvmlDeviceGetCount()
        logger.warning(f'Start to find the GPU device of using memory<{threshold_mem/1000000}M ...')
        while True:
            for i in range(cnt):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                if info.used < threshold_mem:  # 5G    
                    logger.warning(f'GPU-{i} used {info.used/1000000:.1f} M, so the program will use GPU-{i}.') 
                    return i
            cur_time = time.time()
            during = int(cur_time-start)+1
            if  during % 1800 == 0:
                find_times+=0.5
                logger.warning(f'已经经过{find_times}小时，还未找到可用的GPU。')
            
            if find_times > find_ours: # 如果超过 find_ours 个小时还没有分配到GPU，则停止程序
                logger.warning(f'已经经过{find_times}小时，还未找到可用的GPU，终止程序。')
                exit()
    except:
        return 0
        
        
def set_logger(to_console=True, log_file=None):
    """设置logger输出的位置
    Args:
        to_console (bool, optional): 是否输出到 console. Defaults to True.
        log_file (str, optional): 是否输出到 文件. Defaults to None.
    """
    logger = logging.getLogger()  # 不加名称设置root logger
    level = logging.INFO
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    # 使用FileHandler输出到文件
    if log_file != '' and log_file is not None:
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    # 使用StreamHandler输出到屏幕
    if to_console:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    with contextlib.suppress(Exception):
        import numpy as np
        np.random.seed(seed)
        
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True


def dump_pkl(data, f_name):
    with open(f_name, 'wb') as f:
        pickle.dump(data, f)


def load_pkl(f_name):
    with open(f_name, 'rb') as f:
        return pickle.load(f)


def dump_json(data, f_name):
    with open(f_name, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)


def load_json(f_name):
    with open(f_name, 'r', encoding='utf-8') as fr:
        data = json.load(fr)
    return data

def set_to_orderedlist(set_obj):
    """
    将set转为list并进行排序
    """
    return sorted(list(set_obj))

def set_to_dict(set_obj):
    """
    将传进来的集合转成list并排序后，再转为字典
    list_obj = sorted(list(set_obj))
    return {v:idx for idx,v in enumerate(list_obj)}
    """
    list_obj = sorted(list(set_obj))
    return {v:idx for idx,v in enumerate(list_obj)}

    

def get_entity_bio(seq, id2label):
    """Gets entities from sequence.
    note: BIO
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        get_entity_bio(seq)
        #output
        [['PER', 0, 1], ['LOC', 3, 3]]
    from ark_nlp
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
            chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx

            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks

def convert_span_labeled_2_bio(span_labeled_file_or_data, sep = ' ', write_file = None, num_sentences=-1):
    """将span标注转换为bio标注
    Args:
        span_labeled_file_or_data (str或list): 要转化的数据
        sep (str, optional): _description_. Defaults to ' '.
        write_file (None, str): 写入的文件路径，如果为None，则不写入文件.
        num_sentences (int): 转换多少个句子，-1代表全部转换
    Return:
        bio_str (str): 转换好的字符串，可用于直接写入文件里
    """
    if isinstance(span_labeled_file_or_data,str):
        span_labeled_data = load_json(span_labeled_file_or_data)
    else:
        span_labeled_data = span_labeled_file_or_data
    predict_results = []
    convert_num = 0
    for data in span_labeled_data:
        _line = data['text']
        label = len(_line) * ['O']
        for _preditc in data['labels']:
            if 'I' in label[_preditc['start_idx']]:
                continue
            if 'B' in label[_preditc['start_idx']] and 'O' not in label[_preditc['end_idx']]:
                continue
            if 'O' in label[_preditc['start_idx']] and 'B' in label[_preditc['end_idx']]:
                continue
            label[_preditc['start_idx']] = 'B-' +  _preditc['type']
            label[_preditc['start_idx']+1: _preditc['end_idx']+1] = (_preditc['end_idx'] - _preditc['start_idx']) * [('I-' +  _preditc['type'])]
        predict_results.append([_line, label])
        convert_num+=1
        if num_sentences>=0 and convert_num>=num_sentences:
            break
    bio_str = ''
    for _result in predict_results:
        for word, tag in zip(_result[0], _result[1]):
            bio_str += f'{word}{sep}{tag}\n'
        bio_str += '\n'
    with open(write_file, 'w', encoding='utf-8') as f:
        f.write(bio_str)
        
    return bio_str

def convert_bio_2_span_labeled(bio_file_path, sep=' ', write_file = None, num_sentences=-1, with_labels=True):
    """读取BIO格式的文件到datalist中
    Args:
        bio_file_path (str): BIO文件路径。
        sep (str, optional): 字符和标签之间的分隔符. Defaults to ' '。
        write_file (str, None): 结果输出的文件路径，None则不输出到文件。
    Returns:
        list[dict]: 返回值
    Examples:
        bio_file 内容格式是 '我 O\n爱 O\n中 B-0\n国 I-0'，即文件包含两列，左列为字符，右列为标签。句子与句子之间用一个空行分隔。
        读取之后得到:
        [{'text':我爱中国, 'labels':[{'start_idx':2,'end_idx':3,'type':0,'entity':'中国'}]},
         ......
        ]
    """
    datalist = []
    convert_num = 0
    with open(bio_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines.append('\n')
        text = []
        labels = []
        if with_labels:
            for line in lines: 
                if line == '\n':    
                    text = ''.join(text)
                    entity_labels = []
                    for _type, _start_idx, _end_idx in get_entity_bio(labels, id2label=None):
                        entity_labels.append({
                            'start_idx': _start_idx,
                            'end_idx': _end_idx,
                            'type': _type,
                            'entity': text[_start_idx: _end_idx+1]
                        })

                    if not text:
                        continue
                    
                    datalist.append({
                        'text': text,
                        'labels': entity_labels
                    })
                    convert_num+=1
                    if num_sentences>=0 and convert_num>=num_sentences:
                        break
                    text = []
                    labels = []

                elif line == f' {sep}O\n':
                    text.append(' ')
                    labels.append('O')
                else:
                    line = line.strip().split(sep=sep)
                    if len(line) == 1:
                        term = ' '
                        label = line[0]
                    else:
                        term, label = line
                    text.append(term)
                    labels.append(label)
        else:
            for line in lines:
                if line == '\n':
                    text = ''.join(text)
                    entity_labels = []
                    if not text:
                        continue
                    
                    datalist.append({
                        'text': text,
                        'labels': entity_labels
                    })
                    convert_num+=1
                    if num_sentences>=0 and convert_num>=num_sentences:
                        break
                    text = []
                else:
                    line = line.strip()
                    if len(line) == 0:
                        term = ' '
                    else:
                        term = line
                    text.append(term)
                    
    if write_file:
        dump_json(datalist,write_file)
    return datalist


def get_tensor_device(tensor):
    """获取输入参数的设备（cpu，cuda:0, cuda:1, ...)

    Args:
        tensor (Tensor): 

    Returns:
        str: 返回输入Tensor的设备
    """
    try:
        device = tensor.get_device()
        if -1 == device:
            return 'cpu'
        else:
            return f'cuda:{device}'
    except:
        return 'cpu'
    
    
def token_index2char_index(offsets, pad_token_idx = 0):
    """根据token下标到单词下标的映射，推导单词对应的token位置
    TODO: 如果某个字符没有对应的token怎么办？

    Args:
        offsets (List[Touple]): token下标到单词下标的映射

    Returns:
        dict: 单词对应的token位置
    """
    char_no_token = set()
    word_idx2token_idx = {}
    pre_char_idx = 0
    for token_idx, word_pos in enumerate(offsets):
        s, t = word_pos # token 对应的 word 起始和结束位置
        if s != pre_char_idx:
            for i in range(pre_char_idx, s):
                word_idx2token_idx[s] = pad_token_idx
        if t-s == 0: # 特殊的token，无对应的字符
            continue
        if t-s == 1: # 一个token对应一个word，反过来一个word对应一个token
            word_idx2token_idx[s] = (token_idx, token_idx+1)
        else: # 一个token对应多个word，反过来每个word都对应同一个token
            for i in range(s,t):
                word_idx2token_idx[i] = (token_idx, token_idx+1)
                
    return word_idx2token_idx