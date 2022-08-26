# -*- encoding: utf-8 -*-
'''
@File    :   Datasets.py
@Time    :   2022/04/18 09:01:31
@Author  :   Yuan Wind
@Desc    :   None
'''
from collections import defaultdict
import logging
from typing import Dict,List

from scripts.utils import token_index2char_index
logger = logging.getLogger(__name__.replace('_', ''))
from torch.utils.data import Dataset
import torch
import os
from transformers import PreTrainedTokenizer

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
        self.all_features = self.convert_to_features(insts) if convert_here else None
    
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

class RRGDataset(BaseDataset):
    def __init__(self, config, insts, tokenizer, vocab, data_type='train', convert_here=False):
        super().__init__(config, insts, tokenizer, vocab, data_type, convert_here)
        
    @staticmethod
    def convert_to_features(config, insts, tokenizer:PreTrainedTokenizer, vocab):
        """将数据转换为模型输入的参数

        Args:
            config (_type_): _description_
            insts (_type_): _description_
            tokenizer (PreTrainedTokenizer): _description_
            vocab (_type_): _description_

        Returns:
            features: 对应于模型输入的参数，需要转为tensor
        """
        cls_id = tokenizer.cls_token_id
        sep_id = tokenizer.sep_token_id
        pad_id = tokenizer.pad_token_id
        features = {
            'src_input_ids': [],
            'tgt_input_ids':[],
            'pi_input_ids':[],
            'src_attention_mask': [],
            'tgt_attention_mask': [],
            'pi_attention_mask':[],
            'bat_triple_idxs':[]
        }
        src_text_max_len = config.max_seq_len
        for idx, inst in enumerate(insts):
            src_tokenized = tokenizer(inst['src'], add_special_tokens=False)
            tgt_tokenized = tokenizer(inst['tgt'], add_special_tokens=True)
            pi_tokenized = tokenizer(inst['properties'], add_special_tokens=True)
            src_input_ids = [cls_id] + src_tokenized['input_ids'][:src_text_max_len] + [sep_id]
            
            if config.add_triples:
                
                triples_sep_idxs = defaultdict(list)
                for triple in inst['triples']:
                    aspect_words = triple.split('__')[0]
                    aspect = vocab.token2aspect.get(aspect_words,'[UNK]')
                    triples_sep_idxs[aspect].append(len(src_input_ids)-1)
                    triple = f'{aspect}__{triple}'
                    t_t = tokenizer(triple, add_special_tokens=False)
                    src_input_ids += t_t['input_ids'] + [sep_id]
                    if len(src_input_ids) > 512:
                        src_input_ids = src_input_ids[:512]
                        break
                triples_sep_idxs = list(triples_sep_idxs.values())
                features['bat_triple_idxs'].append(triples_sep_idxs)
            src_attention_mask = [1] * len(src_input_ids)
            features['src_input_ids'].append(src_input_ids)
            features['tgt_input_ids'].append(tgt_tokenized['input_ids'])
            features['pi_input_ids'].append(pi_tokenized['input_ids'])
            features['src_attention_mask'].append(src_attention_mask)
            features['tgt_attention_mask'].append(tgt_tokenized['attention_mask'])
            features['pi_attention_mask'].append(pi_tokenized['attention_mask'])
            
        for k,v in features.items():
            _, new_v = batch_padding(v, padding='longest', padding_id=pad_id)
            if k == 'bat_triple_idxs':
                for idx in range(len(new_v)):
                    tmp = []
                    for i in new_v[idx]:
                        if type(i) == int:
                            tmp.append([i])
                        else:
                            tmp.append(i)
                    new_v[idx] = tmp
                features[k] = new_v
                continue # 不用转成tensor类型
            features[k] = torch.tensor(new_v,dtype=torch.long)

        return features
    
def batch_padding(inputs:List[int], padding='longest', max_length = None, padding_id = 0, padding_side = 'right'):
    """将一个batch的数据padding到同样长度

    Args:
        inputs (List): 待padding的数据
        padding (str, optional): padding的策略， ['longest', 'max_length']. Defaults to 'longest'.
        max_length(int, optional): 如果 padding=='max_length', 则padding到指定的长度。
        padding_id (int, optional): padding的id. Defaults to 0.
        padding_side (str, optional): padding的方向， ['left','right']. Defaults to 'left'.
    """
    real_length = [len(i) for i in inputs]
    if padding == 'longest':
        max_length = max(real_length)
    padding_ids = [[padding_id]*(max_length-i) for i in real_length]
    if padding_side == 'right':
        padded = [inputs[idx]+padding_ids[idx] for idx in range(len(inputs))]
    else:
        padded = [padding_ids[idx]+inputs[idx] for idx in range(len(inputs))]
    return real_length, padded

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
            labels = torch.ones(config.max_seq_len, config.max_seq_len).long() * vocab.label2id['PAD']
            labels_symmetry = torch.ones(config.max_seq_len, config.max_seq_len).long() * vocab.label2id['PAD']
            word_pair_position, word_pair_deprel, word_pair_pos, word_pair_synpost = \
                [torch.zeros(config.max_seq_len, config.max_seq_len).long() for _ in range(4)] 
            
            
            token_range = token_index2char_index(token_features[idx].offsets)
            length = len(token_range)
            for triple in inst['triples']:
                aspect = triple['target_tags']
                opinion = triple['opinion_tags']
                aspect_span = get_spans(aspect)
                opinion_span = get_spans(opinion)
                '''set tag for aspect'''
                for l, r in aspect_span:
                    start = token_range[l][0]
                    end = token_range[r][1]
                    for i in range(start, end+1):
                        for j in range(i, end+1):
                            if j == start:
                                labels[i][j] = vocab.label2id['B-A']
                            elif j == i:
                                labels[i][j] = vocab.label2id['I-A']
                            else:
                                labels[i][j] = vocab.label2id['A']

                    for i in range(l, r+1):
                        set_tag = 1 if i == l else 2
                        al, ar = token_range[i]
                        '''mask positions of sub words'''
                        labels[al+1:ar+1, :] = vocab.label2id.get('PAD')
                        labels[:, al+1:ar+1] = vocab.label2id.get('PAD')

                '''set tag for opinion'''
                for l, r in opinion_span:
                    start = token_range[l][0]
                    end = token_range[r][1]
                    for i in range(start, end+1):
                        for j in range(i, end+1):
                            if j == start:
                                labels[i][j] = vocab.label2id['B-O']
                            elif j == i:
                                labels[i][j] = vocab.label2id['I-O']
                            else:
                                labels[i][j] = vocab.label2id['O']

                    for i in range(l, r+1):
                        set_tag = 1 if i == l else 2
                        pl, pr = token_range[i]
                        labels[pl+1:pr+1, :] = vocab.label2id.get('PAD')
                        labels[:, pl+1:pr+1] = vocab.label2id.get('PAD')

                for al, ar in aspect_span:
                    for pl, pr in opinion_span:
                        for i in range(al, ar+1):
                            for j in range(pl, pr+1):
                                sal, sar = token_range[i]
                                spl, spr = token_range[j]
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
                start, end = token_range[i][0], token_range[i][1]
                for j in range(length):
                    s, e = token_range[j][0], token_range[j][1]
                    for row in range(start, end + 1):
                        for col in range(s, e + 1):
                            word_pair_position[row][col] = vocab.post2id.get(abs(row - col), vocab.post2id.get('UNK'))
            
            """2. generate deprel index of the word pair"""
            for i in range(length):
                start = token_range[i][0]
                end = token_range[i][1]
                for j in range(start, end + 1):
                    s, e = token_range[head[i] - 1] if head[i] != 0 else (0, 0)
                    for k in range(s, e + 1):
                        word_pair_deprel[j][k] = vocab.deprel2id.get(deprel[i])
                        word_pair_deprel[k][j] = vocab.deprel2id.get(deprel[i])
                        word_pair_deprel[j][j] = vocab.deprel2id.get('self')
            
            """3. generate POS tag index of the word pair"""
            for i in range(length):
                start, end = token_range[i][0], token_range[i][1]
                for j in range(length):
                    s, e = token_range[j][0], token_range[j][1]
                    for row in range(start, end + 1):
                        for col in range(s, e + 1):
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
                start, end = token_range[i][0], token_range[i][1]
                for j in range(length):
                    s, e = token_range[j][0], token_range[j][1]
                    for row in range(start, end + 1):
                        for col in range(s, e + 1):
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


class NERDataset1(BaseDataset):
    def __init__(self, config, insts, tokenizer, vocab, data_type='train', convert_here=False):
        super().__init__(config, insts, tokenizer, vocab, data_type, convert_here)
        
    @staticmethod
    def convert_to_features(config, insts, tokenizer:PreTrainedTokenizer, vocab):
        features = []
        # sentences1 = []
        # sentences2 = []
        return_offsets_mapping = True
        # for inst in insts:
        #     sentences1.append(inst['tokens'])
            # sentences2.append(' '.join(inst['tokens']))
        
        
        # 经过测试，下面的tokenized_outs1.input_ids==tokenized_outs2.input_ids, 
        # 即is_split_into_words 原理就是使用空格先进行拼接，再tokenize。 但是offset_mapping不同，前者的offset_mapping全为1
        # tokenized_outs1 = tokenizer(sentences1, is_split_into_words=True, add_special_tokens=True, 
        #                               padding = 'longest', return_offsets_mapping = return_offsets_mapping, 
        #                               return_tensors = 'pt') 
        
        # tokenized_outs2 = tokenizer(sentences2, add_special_tokens=True, 
        #                               padding = 'longest', return_offsets_mapping = return_offsets_mapping, 
        #                               return_tensors = 'pt')

        

        blank_token_id = tokenizer.convert_tokens_to_ids('▁')
        bat_max_len = 0
        bat_labels = []
        for idx, inst in enumerate(insts):
            tokenized_outs1 = tokenizer(inst['tokens'], is_split_into_words=True, add_special_tokens=True, 
                                      return_offsets_mapping = return_offsets_mapping, 
                                      return_tensors = 'pt') 
            non_blank = tokenized_outs1.input_ids[0].ne(blank_token_id)
            input_ids = tokenized_outs1.input_ids[0][non_blank]
            attention_mask = tokenized_outs1.attention_mask[0][non_blank]
            token_type_ids = tokenized_outs1.token_type_ids[0][non_blank]
            
            # 构建labels
            # 初始化labels 的shape为[seq_len], 值为 -100， -100在计算损失时会被忽略。
            labels = torch.ones_like(input_ids) * -100
            tokenids2wordidx = [i for idx, i in enumerate(tokenized_outs1[0].word_ids) if non_blank[idx]]
            exist_word_tags = set() # 每个word的第一个赋予tag，该word其他的tokens不给tag
            for token_idx, word_idx in enumerate(tokenids2wordidx):
                # 非 blank_token_id 才给对应的标签，否则就是默认值-100，计算loss会忽略掉
                if word_idx is not None and input_ids[token_idx]!=blank_token_id \
                    and word_idx not in exist_word_tags:
                    ori_tag = inst['tags'][word_idx]
                    labels[token_idx] = vocab.labels2id.get(ori_tag, 'O')
                    exist_word_tags.add(word_idx)
            
            # ids->tokens: a = tokenizer.convert_ids_to_tokens(tokenized_outs1.input_ids[idx], skip_special_tokens=True)
            # tokens->string: res = tokenizer.convert_tokens_to_string(a).replace(' ', '')
            one_feature = {
                'input_ids':input_ids,
                'attention_mask':attention_mask,
                'token_type_ids':token_type_ids,
            }
            bat_labels.append(labels)
            features.append(one_feature)
            bat_max_len = max(bat_max_len,len(input_ids))
            
        # Padding batch
        # for feature in features:
        tokenizer.pad(features, padding='longest', return_tensors='pt')
        return features, bat_labels
    
    @staticmethod
    def convert_to_features_bak(config, insts, tokenizer, vocab):
        import jieba
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



class NERDataset(BaseDataset):
    def __init__(self, config, insts, tokenizer, vocab, data_type='train', convert_here=False):
        super().__init__(config, insts, tokenizer, vocab, data_type, convert_here)
        
    
    @staticmethod
    def convert_to_features(config, insts, tokenizer, vocab):
        import jieba
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
    