# -*- encoding: utf-8 -*-
'''
@File    :   Datasets.py
@Time    :   2022/04/18 09:01:31
@Author  :   Yuan Wind
@Desc    :   None
'''
from collections import defaultdict
import logging
from typing import Dict
from tqdm import tqdm
from scripts.utils import token_index2char_index
logger = logging.getLogger(__name__.replace('_', ''))
from torch.utils.data import Dataset
import torch
import os
import jieba
from scripts.evaluater import find_triplet

class BaseDataset(Dataset):
    def __init__(self, config, insts, tokenizer = None, vocab = None, data_type = 'train', convert_here = False):
        """所有数据集的基类

        Args:
            config (Config): 包含各种配置信息和超参定义等
            insts (list): 原始数据
            tokenizer (optional): tokenizer初始化. Defaults to None.
            vocab (optional): vocab初始化. Defaults to None.
            data_type (str, optional): [train, dev, test],标志着训练集、验证集和测试集. Defaults to 'train'.
            convert_here (bool, optional): 是否在初始化类的时候就把全部的原始数据转化为特征. Defaults to False.
        """
        super().__init__()
        self.config = config
        self.data_type = data_type  # 区分该数据集是训练集、验证集和测试集
        self.insts = insts
        self.tokenizer = tokenizer
        self.vocab = vocab
        # 将inst转化为特征后保存到features中，类型为List[Dict]
        self.all_features = self.convert_to_features(config, insts, tokenizer, vocab) if convert_here else None
    
    @property
    def items(self):
        return self.all_features if self.all_features is not None else self.insts
      
    def __len__(self):
        """
        如果子类不实现该方法，那么Trainer里边默认的Dataloader将不会进行sample，否则会进行randomsample
        TODO 不实现的话有点问题
        """
        return len(self.items)
    
    def __getitem__(self,index) :
        """
        请与Datacollator进行联合设计出模型的batch输入，建议利用字典进行传参。
        """
        return self.items[index]
    
    @staticmethod
    def convert_to_features(config, insts, tokenizer, vocab):
        """批量将每个inst转化为输入到模型的特征（参数）

        Args:
            config (MyConfig): 参数
            insts (list): 待转化的原始数据
            tokenizer (Tokenizer): tokenizer
            vocab (BaseVocab): vocab
        Returns:
            features(List[Dict]): 转化后的特征list
        """
        raise NotImplementedError(f"convert_to_features(): not implemented!")

class ASTEDataset(BaseDataset):
    def __init__(self, config, insts, tokenizer, vocab, data_type='train', convert_here=False):
        """方面-意见-情感三元组抽取数据集
        """
        super().__init__(config, insts, tokenizer, vocab, data_type, convert_here)
        
    @staticmethod
    def convert_to_features(config, insts, tokenizer, vocab):
        
        def get_token_pos_dep(tokens, pos, dep, head):
            res_p, res_d, res_h = [], [], []
            for idx, word in enumerate(tokens):
                for ch in word:
                    res_p.append(pos[idx])
                    res_d.append(dep[idx])
                    res_h.append(head[idx])
            return res_p, res_d, res_h
        
        def get_spans(tags):
            '''for BIO tag'''
            tags = tags.strip().split()
            length = len(tags)
            spans = []
            start = -1
            for i in range(length):
                if tags[i].endswith('B'):
                    if start != -1:
                        spans.append([start, i - 1])
                    start = i
                elif tags[i].endswith('O'):
                    if start != -1:
                        spans.append([start, i - 1])
                        start = -1
            if start != -1:
                spans.append([start, length - 1])
            return spans
        
        sentences = []
        for inst in insts:
            sentences.append(''.join(inst['sentence']))
            
        token_features = tokenizer(sentences, return_offsets_mapping=True, add_special_tokens=True,
                                   padding = 'max_length', max_length = config.max_seq_len, return_tensors = 'pt')
        features = []
        for idx, inst in enumerate(insts):
            # input_ids, attention_mask,token_type_ids, word_pair_position, word_pair_deprel, word_pair_pos, word_pair_synpost
            
            words = inst['sentence']
            chars = ''.join(words)
            labels = torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.long) * vocab.label2id['PAD']
            labels_symmetry = torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.long) * vocab.label2id['PAD']
            word_pair_position, word_pair_deprel, word_pair_pos, word_pair_synpost = \
                [torch.zeros(config.max_seq_len, config.max_seq_len, dtype=torch.long) for _ in range(4)] 
            
            
            token_range = token_index2char_index(token_features[idx].offsets)
            length = len(token_range)
            for triple in inst['triples']:
                aspect = triple['target_tags']
                opinion = triple['opinion_tags']
                aspect_span = get_spans(aspect)
                opinion_span = get_spans(opinion)
                '''set tag for aspect'''
                for l, r in aspect_span:
                    start = token_range.get(l, (0,0))[0]
                    end = token_range.get(r, (0,0))[1]
                    for i in range(start, end+1):
                        for j in range(i, end+1):
                            if j == start:
                                labels[i][j] = vocab.label2id['B-A']
                            elif j == i:
                                labels[i][j] = vocab.label2id['I-A']
                            else:
                                labels[i][j] = vocab.label2id['A']

                    # for i in range(l, r+1):
                    #     set_tag = 1 if i == l else 2
                    #     al, ar = token_range.get(i, (0,0))
                    #     '''mask positions of sub words'''
                    #     labels[al+1:ar+1, :] = vocab.label2id.get('PAD')
                    #     labels[:, al+1:ar+1] = vocab.label2id.get('PAD')

                '''set tag for opinion'''
                for l, r in opinion_span:
                    start = token_range.get(l, (0,0))[0]
                    end = token_range.get(r, (0,0))[1]
                    for i in range(start, end+1):
                        for j in range(i, end+1):
                            if j == start:
                                labels[i][j] = vocab.label2id['B-O']
                            elif j == i:
                                labels[i][j] = vocab.label2id['I-O']
                            else:
                                labels[i][j] = vocab.label2id['O']

                    # for i in range(l, r+1):
                    #     set_tag = 1 if i == l else 2
                    #     pl, pr = token_range.get(i, (0,0))
                    #     labels[pl+1:pr+1, :] = vocab.label2id.get('PAD')
                    #     labels[:, pl+1:pr+1] = vocab.label2id.get('PAD')

                for al, ar in aspect_span:
                    for pl, pr in opinion_span:
                        for i in range(al, ar+1):
                            for j in range(pl, pr+1):
                                sal, sar = token_range.get(i, (0,0))
                                spl, spr = token_range.get(j, (0,0))
                                labels[sal:sar+1, spl:spr+1] = vocab.label2id.get('PAD')
                                if i > j:
                                    labels[spl][sal] = vocab.label2id[triple['sentiment']]
                                else:
                                    labels[sal][spl] = vocab.label2id[triple['sentiment']]

            for i in range(1, length-1):
                for j in range(i, length-1):
                    labels_symmetry[i][j] = labels[i][j]
                    labels_symmetry[j][i] = labels_symmetry[i][j]
            
            postag, deprel, head = get_token_pos_dep(words, inst['postag'], inst['deprel'], inst['head'])
            '''1. generate position index of the word pair'''
            for i in range(length):
                start, end = token_range.get(i, (0,0))[0], token_range.get(i, (0,0))[1]
                for j in range(length):
                    s, e = token_range.get(j, (0,0))[0], token_range.get(j, (0,0))[1]
                    for row in range(start, end):
                        for col in range(s, e):
                            word_pair_position[row][col] = vocab.post2id.get(abs(row - col), vocab.post2id.get('UNK'))
            
            """2. generate deprel index of the word pair"""
            for i in range(length):
                start = token_range.get(i, (0,0))[0]
                end = token_range.get(i, (0,0))[1]
                for j in range(start, end):
                    s, e = token_range.get(head[i] - 1, (0,0)) if head[i] != 0 else (0, 0)
                    for k in range(s, e):
                        word_pair_deprel[j][k] = vocab.deprel2id.get(deprel[i])
                        word_pair_deprel[k][j] = vocab.deprel2id.get(deprel[i])
                        word_pair_deprel[j][j] = vocab.deprel2id.get('self')
            
            """3. generate POS tag index of the word pair"""
            for i in range(length):
                start, end = token_range.get(i, (0,0))[0], token_range.get(i, (0,0))[1]
                for j in range(length):
                    s, e = token_range.get(j, (0,0))[0], token_range.get(j, (0,0))[1]
                    for row in range(start, end):
                        for col in range(s, e):
                            word_pair_pos[row][col] = vocab.postag2id.get(tuple(sorted([postag[i], postag[j]])))
                            
            """4. generate synpost index of the word pair"""
            tmp = [[0]*length for _ in range(length)]
            for i in range(length):
                j = head[i]
                if j == 0:
                    continue
                tmp[i][j - 1] = 1
                tmp[j - 1][i] = 1

            tmp_dict = defaultdict(list)
            for i in range(length):
                for j in range(length):
                    if tmp[i][j] == 1:
                        tmp_dict[i].append(j)
                
            word_level_degree = [[4]*length for _ in range(length)]

            for i in range(length):
                node_set = set()
                word_level_degree[i][i] = 0
                node_set.add(i)
                for j in tmp_dict[i]:
                    if j not in node_set:
                        word_level_degree[i][j] = 1
                        node_set.add(j)
                    for k in tmp_dict[j]:
                        if k not in node_set:
                            word_level_degree[i][k] = 2
                            node_set.add(k)
                            for g in tmp_dict[k]:
                                if g not in node_set:
                                    word_level_degree[i][g] = 3
                                    node_set.add(g)
            
            for i in range(length):
                start, end = token_range.get(i, (0,0))[0], token_range.get(i, (0,0))[1]
                for j in range(length):
                    s, e = token_range.get(j, (0,0))[0], token_range.get(j, (0,0))[1]
                    for row in range(start, end):
                        for col in range(s, e):
                            word_pair_synpost[row][col] = vocab.syn_post2id.get(word_level_degree[i][j], vocab.syn_post2id['UNK'])
        
                
            one_feature = {
                'input_ids':torch.tensor(token_features[idx].ids),
                'attention_mask':torch.tensor(token_features[idx].attention_mask),
                'token_type_ids':torch.tensor(token_features[idx].type_ids),
                'word_pair_position':word_pair_position,
                'word_pair_deprel':word_pair_deprel,
                'word_pair_pos':word_pair_pos,
                'word_pair_synpost':word_pair_synpost,
                'labels': labels,
                'labels_symmetry': labels_symmetry
            }
            features.append(one_feature)
        
        
        return features



class NERDataset(BaseDataset):
    def __init__(self, config, insts, tokenizer, vocab, data_type='train', convert_here=False):
        super().__init__(config, insts, tokenizer, vocab, data_type, convert_here)
        
    
    @staticmethod
    def convert_to_features(config, insts, tokenizer, vocab):
        features = []
        # TODO 改成 tokenize-fast版本
        for inst in insts:
            words = jieba.cut(inst['text'])
            tokens = tokenizer.tokenize(inst['text'])[:config.max_seq_len-2]
            token_mapping = tokenizer.get_token_mapping(inst['text'], tokens)  # token_id -> word_idxs
            pad_star_idx = len(tokens) +1 # 加上一个cls的token
            word_idx_2_token_idx = {}
            for token_id, word_idxs in enumerate(token_mapping):
                for w_id in word_idxs:
                    word_idx_2_token_idx[w_id] = token_id
            # tokens = []
            entity_tpyes = {} # token_id --> types
            w_id = 0
            for word in words:
                for char in word:
                    token_id = word_idx_2_token_idx.get(w_id)
                    if token_id is not None:
                        entity_tpyes[token_id] = vocab.entity2types.get(word,[])
                    # else:
                    #     logger.info(f'Ignored char {char}')
                    w_id += 1


            start_mapping = {j[0]: i for i, j in enumerate(token_mapping) if j}
            end_mapping = {j[-1]: i for i, j in enumerate(token_mapping) if j}

            input_ids = tokenizer.sequence_to_ids(tokens)

            input_ids, input_mask, segment_ids = input_ids
            
            # 全部初始化为 O 对应的id
            global_label = torch.zeros((vocab.num_labels,tokenizer.max_seq_len,tokenizer.max_seq_len))
            
            
            # 只做BIO分词任务的标签， 全部初始化为 O 对应的id
            bio_label = torch.tensor([vocab.bio2id['O']]*tokenizer.max_seq_len,dtype=torch.long)
            
            for info_ in inst['labels']:
                if info_['start_idx'] in start_mapping and info_['end_idx'] in end_mapping:
                    start_idx = start_mapping[info_['start_idx']]
                    end_idx = end_mapping[info_['end_idx']]
                    if start_idx > end_idx or info_['entity'] == '':
                        continue
                    # 构建 Global Pointer 的标签
                    tp = vocab.label2id[info_['type']]
                    global_label[tp, start_idx+1, end_idx+1] = 1
                    # 构建 BIO 标签
                    bio_label[start_idx+1] = vocab.bio2id['B'] # start_idx加 1 是因为前面多了一个 cls token
                    if end_idx-start_idx > 0: # 如果实体多于一个token，就把除第一个token以外的都标为 I
                        bio_label[start_idx+2:end_idx+2] = vocab.bio2id['I']
                    # bio_label[start_idx+1] = vocab.bio2id['B']
                    # if start_idx+2<=end_idx+1:
                    #     bio_label[start_idx+2:end_idx+2] = vocab.bio2id['I']
                # else:
                    # logger.info(f'Ignored {info_} in "{inst['text']}"')
                    
            # 填充pad标签
            # global_label[:,pad_star_idx:,pad_star_idx:] = vocab.label2id['PAD'] 
            bio_label[pad_star_idx:] = vocab.bio2id['PAD'] 
            
            fe = {
                'input_ids': torch.tensor(input_ids),
                'attention_mask': torch.tensor(input_mask),
                'token_type_ids': torch.tensor(segment_ids),
                'label_ids': global_label,
                }
            if config.add_type_emb:
                type_ids = [ [vocab.label2id['PAD']]*15 ] 
                for token_id in range(len(tokens)):
                    types = entity_tpyes[token_id]
                    t_ids = [vocab.label2id.get(tp) for tp in types] + [vocab.label2id['PAD']]
                    t_ids = t_ids+[vocab.label2id['PAD']] * (15-len(t_ids))
                    type_ids.append(t_ids)
                type_ids.extend([[vocab.label2id['PAD']] * 15 for _ in range(len(input_ids)-len(type_ids))])
                fe['type_ids'] = torch.tensor(type_ids)

            if config.mtl_cls_w>0:
                cls_label = torch.tensor([vocab.cls2id[vocab.title2type.get(inst['text'],'UNK')]],dtype=torch.long)
                fe['cls_label'] = cls_label
            
            if config.mtl_bio_w>0:
                fe['bio_label'] = bio_label
                
            
            features.append(fe)
        
        return features



    


class MyLineByLineTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(self, file_path: str):
        if os.path.isfile(file_path) is False:
            raise ValueError(f"Input file path {file_path} not found")

        logger.info(f"Creating features from dataset file at {file_path}")

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        examples = lines

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]
    
    # def __len__(self):
    #     return len(self.insts)
    
    # def __getitem__(self,index):
    #     return self.items[index]
    