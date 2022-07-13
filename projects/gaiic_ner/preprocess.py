# -*- encoding: utf-8 -*-
'''
@File    :   DataUtils.py
@Time    :   2022/04/18 09:01:43
@Author  :   Yuan Wind
@Desc    :   None
'''
import logging
logger = logging.getLogger(__name__.replace('_', ''))
import sys
sys.path.extend(['../../','../','./','code'])

from collections import defaultdict
from scripts.utils import convert_bio_2_span_labeled, dump_json, dump_pkl, sort_dict
from modules.data.vocabs import TokenVocab
from modules.data.datasets import NERDataset

def anysis_data(datalist): 
    label2cnt = defaultdict(int)
    for d in datalist: 
        # text = d['text']
        lbls = d['labels']
        for lbl in lbls: 
            label2cnt[lbl['type']] +=1
    return label2cnt

def split_data(datalist, percent = 0.90):
    """训练集和验证集的标签类型尽量同分布。首先保证每个实体标签在验证集中出现一次，
    然后根据设定的percent来决定是否往验证集中添加数据
    """
    # 
    label2cnt = anysis_data(datalist)
    tmp = sort_dict(label2cnt, 'v')
    x,y = [i[0] for i in tmp], [i[1] for i in tmp]
    for a,b in tmp: 
        print(f'{a}:{b}')
    # plot_fig(x,y)
    
    dev_cnt = defaultdict(int)
    
    train, dev = [], []
    for lbl, all_cnt in label2cnt.items():
        dev_cnt[lbl] = int(all_cnt*(1-percent))
        
    dev_lbl_cnt = sort_dict(dev_cnt, 'v')
    cur_cnt = defaultdict(int)
    
    is_in_dev = {}
    for idx, d in enumerate(datalist): 
        lbls = d['labels']
        is_in_dev[idx] = False
        for lbl in lbls:
            l = lbl['type']
            if l not in cur_cnt.keys() and not is_in_dev[idx]:
                dev.append(d)
                is_in_dev[idx] = True
            if is_in_dev[idx]: 
                cur_cnt[l] +=1
    
    from tqdm import tqdm
    for t_l, t_n in tqdm(dev_lbl_cnt):
        for idx, d in enumerate(datalist): 
            add_in_dev = False
            lbls = d['labels']
            for lbl in lbls:
                l = lbl['type']
                if l==t_l and cur_cnt[t_l]<t_n and not is_in_dev[idx]:
                    dev.append(d)
                    is_in_dev[idx] = True
                    add_in_dev = True
                if add_in_dev: 
                    cur_cnt[l] +=1
        
    for idx, d in enumerate(datalist): 
        if is_in_dev[idx] == False:
            train.append(d)

    
    
    
    dump_json(train, f'data_gaiic_wind/temp_data/{percent:.2f}_train.json')
    dump_json(dev, f'data_gaiic_wind/temp_data/{percent:.2f}_dev.json')
    dump_pkl(train, f'data_gaiic_wind/temp_data/{percent:.2f}_train.pkl')
    dump_pkl(dev, f'data_gaiic_wind/temp_data/{percent:.2f}_dev.pkl')
    return f'percent:{percent}, train_num:{len(train)}, dev_num:{len(dev)}, total:{len(datalist)}\n'

def build_vocab(train_files,dev_files,test_file,save_path='data_gaiic_wind/temp_data/vocab.pkl'):
    vocab = TokenVocab()
    vocab.read_files(train_files,file_type= 'train')
    vocab.read_files(dev_files,file_type= 'dev')
    vocab.read_files(test_file,file_type= 'test')
    vocab.build(vocab.train_insts+vocab.dev_insts) # 使用哪些数据构建 Vocab
    dump_pkl(vocab, save_path)

def train_text_label():
    datalist = convert_bio_2_span_labeled('data_gaiic_wind/train_data/train.txt')
    texts = []
    labels = []
    for data in datalist:
        text = data['text']
        words = []
        label = []
        pre_idx = 0
        for lbls in  data['labels']:
            start = lbls['start_idx']
            end = lbls['end_idx']
            tp = lbls['type']
            if pre_idx < start:
                words.append(text[pre_idx:start])
            words.append(text[start:end+1])
            pre_idx = end+1
            label.append(tp)
        if pre_idx < len(text):
            words.append(text[pre_idx:])
        assert ''.join(words) == text, f'{words}\n{text}'
        texts.append('\t'.join(words))
        labels.append('\t'.join(label))
    f_text = open('data_gaiic_wind/train_data/train_text.txt', 'w', encoding='utf-8')
    f_label = open('data_gaiic_wind/train_data/train_label.txt', 'w', encoding='utf-8')
    f_text.write('\n'.join(texts))
    f_label.write('\n'.join(labels))
    f_text.close()
    f_label.close()
    
def k_fold_split(k, data):
    import random
    random.seed(666)
    random.shuffle(data)
    step = int(len(data) / k) + 1
    split_data =[data[i:i+step]  for i in range(0,len(data),step)]
    print(f'{k} 份长度分别为：{[len(i) for i in split_data]}')
    for idx, one_data in enumerate(split_data):
        dump_json(one_data,f'data_gaiic_wind/temp_data/kfold_data/data_{idx}.json')
    
    return split_data

def build_102w_test_data(files):
    data = []
    f_w = open('data_gaiic_wind/train_data/unlabel_data_102w.txt','a+',encoding='utf-8')
    for one in files:
        with open(one, 'r',encoding='utf-8') as f:
            for line in f:
                f_w.write(line)
                text = line.strip()
                labels = []
                data.append({'text':text, 'labels':labels})
    dump_json(data,'data_gaiic_wind/temp_data/unlabel_data_102w.json')

if __name__ == "__main__":
    
    build_102w_test_data(
        [
        'data_gaiic_wind/train_data/unlabeled_train_data.txt',
        'data_gaiic_wind/train_data/preliminary_test_a/sample_per_line_preliminary_A.txt',
        'data_gaiic_wind/train_data/preliminary_test_b/sample_per_line_preliminary_B.txt'
        ]
    )
    
    # datalist = convert_bio_2_span_labeled('data_gaiic_wind/train_data/train.txt')
    # 划分验证集并生成 vocab
    # dump_json(datalist[:36000],f'data_gaiic_wind/temp_data/data_train.json')
    # dump_json(datalist[36000:],f'data_gaiic_wind/temp_data/data_dev.json')
    # train_files = ['data_gaiic_wind/temp_data/data_train.json']
    # dev_files = ['data_gaiic_wind/temp_data/data_dev.json']
    # test_file = []
    # save_path = f'data_gaiic_wind/temp_data/vocabs/vocab.pkl'
    # build_vocab(train_files,dev_files,test_file,save_path)
    
    # 划分 10 折数据
    # k_fold_split(10, datalist)
    # 生成 10折的 vocab
    # data_dir = '/home/mw/input/train_dev7893/temp_data/temp_data'
    # data_dir = 'data_gaiic_wind/temp_data/kfold_data'
    # for i in range(10):
    #     trains = list(range(10))
    #     trains.remove(i)
    #     train_files = [f'{data_dir}/data_{j}.json' for j in trains]
    #     dev_files = [f'{data_dir}/data_{i}.json']
    #     test_file = []
    #     save_path = f'data_gaiic_wind/temp_data/vocabs/vocab_{i}.pkl'
    #     build_vocab(train_files,dev_files,test_file,save_path)
    
    # 其他方法划分验证集
    # percent = 0.9
    # f_out = open('data_gaiic_wind/temp_data/num.txt', 'w')
    # while percent < 0.98:
    #     f_out.write(split_data(datalist, percent = percent))
    #     percent += 0.01
    # f_out.close()
    
    
    # 加入伪标
    # pseudo_file = '/home/mw/input/track2_public_9711/Tacos AI/public_data/41694_labels.txt'
    # pseudo_data = convert_bio_2_span_labeled(pseudo_file,with_labels=False)
    # dump_json(pseudo_data,'data_gaiic_wind/temp_data/pseudo_data.json')
    
    
