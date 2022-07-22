# -*- encoding: utf-8 -*-
'''
@File    :   DataVocab.py
@Time    :   2022/04/18 09:01:51
@Author  :   Yuan Wind
@Desc    :   None
'''
from collections import Counter, defaultdict
import json
import logging
logger = logging.getLogger(__name__.replace('_', ''))
from scripts.utils import load_json, load_pkl, set_to_orderedlist
from  functools  import  wraps 

def add_labels(label_names=[]):
    """装饰器，为类动态加入 {label_name}2id 和 id2{label_name} 属性。

    Args:
        label_names (list, optional): 待加入的label_names. Defaults to [].
    """
    
    def set_class(vocab_class):
        
        class wrapper(vocab_class):
            def __init__(self):
                super().__init__()
                for name in label_names:
                    label2id = {'PAD':0, 'UNK':1}
                    id2label = ['PAD', 'UNK']
                    self.__setattr__(f'{name}2id', label2id)
                    self.__setattr__(f'id2{name}', id2label)
        return wrapper
    return set_class
            

class BaseVocab:
    def __init__(self, names = []):
        self.all_insts = []
        self.train_insts = []
        self.dev_insts = []
        self.test_insts = []
        for name in names:
            label2id = {'PAD':0, 'UNK':1}
            id2label = ['PAD', 'UNK']
            self.__setattr__(f'{name}2id', label2id)
            self.__setattr__(f'id2{name}', id2label)

    def get_label_num(self, name='labels'):
        val = self.__getattribute__(f'id2{name}')
        return len(val)
    
    def read_files(self, files, file_type = 'train'):
        """
        读取多个文件
        Args:
            files (list[str]): 要读取的文件列表
            file_type (str, optional): ['train','dev','test']. Defaults to 'train'.
        """
        for file in files:
            if file_type == 'train':
                self.train_insts.extend(self.read_file(file))
            elif file_type == 'dev':
                self.dev_insts.extend(self.read_file(file))
            elif file_type == 'test':
                self.test_insts.extend(self.read_file(file))
            self.all_insts.extend(self.read_file(file))
            
    def read_file(self, file_path):
        """读取单个文件的数据
        Args:
            file (str): 文件路径
            file_type (str, optional): ['train','dev','test']. Defaults to 'train'.
        """
        if file_path[-3:] == 'pkl':
            return load_pkl(file_path)
        elif file_path[-3:] == 'son':
            return load_json(file_path)
    
    def set_labels(self, name, labels_set):
        val = self.__getattribute__(f'id2{name}') 
        val += set_to_orderedlist(labels_set)
        self.__setattr__(f'id2{name}', val)
        self.__setattr__(f'{name}2id', {v:idx for idx, v in enumerate(val)})
        
    def build(self):
        """
        根据全部的insts构建vocab
        """
        raise NotImplementedError
    
class RRGVocab(BaseVocab):
    def __init__(self, names = ['labels', 'aspect', 'sentiment']):
        super().__init__(names)
        self.token2aspect = {}
        
    def build(self, aspect2tokens):
        aspect_set = set(aspect2tokens.keys())
        self.set_labels('aspect', aspect_set)
        
        for k, vs in aspect2tokens.items():
            for v in vs:
                self.token2aspect[v] = k    

@add_labels(['labels', 'aspect', 'sentiment'])
class RRGVocab1(BaseVocab):
    def __init__(self):
        super().__init__()
        self.token2aspect = {}
        
    def build(self, aspect2tokens):
        aspect_set = set(aspect2tokens.keys())
        self.set_labels('aspect', aspect_set)
        
        for k, vs in aspect2tokens.items():
            for v in vs:
                self.token2aspect[v] = k             
class ASTEVocab(BaseVocab):
    def __init__(self):
        super().__init__() # Python 3 可以使用直接使用 super().xxx 代替 super(Class, self).xxx
        self.postag2id = {'PAD':0, 'UNK':1}
        self.id2postag = ['PAD', 'UNK']
        
        self.deprel2id = {'PAD':0, 'UNK':1, 'self':2}
        self.id2deprel = ['PAD', 'UNK', 'self']
        
        self.id2label = ['PAD', 'UNK'] + set_to_orderedlist(
            set(['N', 'B-A', 'I-A', 'A', 'B-O', 'I-O', 'O', 'negative', 'neutral', 'positive'])
            )
        self.label2id = {v:idx for idx,v in enumerate(self.id2label)}

        # 位置
        self.post2id = {'PAD':0, 'UNK':1}
        self.id2post = ['PAD', 'UNK']
                
        self.syn_post2id = {'PAD':0, 'UNK':1}
        self.id2syn_post = ['PAD', 'UNK']
        
        self.aspects = defaultdict(int)
    @property
    def num_postag(self):
        return len(self.id2postag)
    
    @property
    def num_deprel(self):
        return len(self.id2deprel)
    
    @property
    def num_post(self):
        return len(self.id2post)
    
    @property
    def num_syn_post(self):
        return len(self.id2syn_post)
        
    def read_file(self, file_path):
        data = load_json(file_path)
        print(f'Read {len(data)} data from {file_path}')
        return data
        

    def build(self):
        postag_set = set()
        deprel_set = set()
        postag_ca = set()
        for inst in self.all_insts:
            postag_ca = postag_ca | set(inst['postag'])
            n = len(inst['postag'])
            tmp_pos = []
            for i in range(n):
                for j in range(n):
                    tup = tuple(sorted([inst['postag'][i], inst['postag'][j]]))
                    tmp_pos.append(tup)
            postag_set = postag_set | set(tmp_pos)
            deprel_set = deprel_set | set(inst['deprel'])
            for triple in inst['triples']:
                if 'aspect' in triple:
                    self.aspects[triple['aspect']] += 1
                    
        self.id2postag += set_to_orderedlist(postag_set)
        self.postag2id = {v:idx for idx, v in enumerate(self.id2postag)}
        
        self.id2deprel += set_to_orderedlist(deprel_set)
        self.deprel2id = {v:idx for idx, v in enumerate(self.id2deprel)}
        
        self.id2post += list(range(128))
        self.post2id = {v:idx for idx, v in enumerate(self.id2post)}
        
        self.id2syn_post += list(range(5))
        self.syn_post2id = {v:idx for idx, v in enumerate(self.id2syn_post)}
        
        print(f'Labels to ID:\n{self.id2label}')
        print(f'Postag to ID:\n{self.id2postag}')
        print(f'Deprel to ID:\n{self.id2deprel}')
        print(f'Aspect info:\n{self.aspects}')
        print(f'Max length :{self.id2post}; Syn_post: 5')

class TokenVocab(BaseVocab):
    def __init__(self):
        super(TokenVocab, self).__init__()
        self.id2bio = ['PAD', 'PAD', 'O','B','I']
        self.bio2id = {'PAD':0, 'UNK':1, 'O':2,'B':3,'I':4}

        self.id2cls = ['PAD', 'UNK']
        self.cls2id = {'PAD':0, 'UNK':1}
        
        self.title2type = {} # 商品标题  ----  二级类目
        self.entity2types = defaultdict(set)  # 实体词 --- 可能的实体类别集合
        
    @property
    def bio_num(self): 
        return len(self.id2bio)
    
    @property
    def cls_num(self): 
        return len(self.id2cls)
    
    
    def read_file(self, file_path):
        if file_path[-3:] == 'pkl':
            return load_pkl(file_path)
        elif file_path[-3:] == 'son':
            return load_json(file_path)
    
    def build(self, insts):
        assert len(insts) != 0, '当前无数据，请先读取数据，再构建Vocab！'
        label_set = set()
        for data in insts:
            labels = data.get('labels')
            for lbl in labels:
                tp = lbl['type']
                entity = lbl['entity']
                label_set.add(tp)
                self.entity2types[entity].add(tp)
        label_set.add('O')
        self.id2label += set_to_orderedlist(label_set)
        self.label2id = {v:idx for idx,v in enumerate(self.id2label)}
        self.build_other_vocab()
        
    def build_other_vocab(self):
        class2id = load_pkl('data_gaiic_wind/train_data/public_data/label2id.pkl')
        id2class = {}
        first_set = set()
        second_set = set()
        third_set = set()
        for k,v in class2id.items():
            if 'None' not in k:
                k = json.loads(k)
                first_set.add(k['first'])
                second_set.add(k['second'])
                third_set.add(k['third'])
            else:
                k = {"first":"None", "second":"None", "third":"None"}
            id2class[v] = k
        assert len(class2id) == len(id2class)
        total_data = load_json('data_gaiic_wind/train_data/public_data/train_datalist2.json')
        cls_set = set()
        for data in total_data:
            title = data['text']
            possible_classid = Counter(data['other_types']).most_common()[0][0]
            possible_class = id2class[possible_classid].get('second')
            self.title2type[title] = possible_class
            cls_set.add(possible_class)
        
        self.id2cls += set_to_orderedlist(cls_set)
        self.cls2id = {v:idx for idx,v in enumerate(self.id2cls)}