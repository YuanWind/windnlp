# -*- encoding: utf-8 -*-
'''
@File    :   DataCollator.py
@Time    :   2022/04/18 09:01:19
@Author  :   Yuan Wind
@Desc    :   None
'''
from collections import defaultdict
from dataclasses import dataclass
import logging
from transformers.data.data_collator  import *
logger = logging.getLogger(__name__.replace('_', ''))
import torch
from typing import Any

class Datacollator:
    def __init__(self,config, vocab, tokenizer = None, convert_method = None, convert_here = True):
        """Datacollator初始化

        Args:
            config (MyConfig): 参数
            vocab (BaseVocab): vocab
            tokenizer (tokenzier, optional): tokenzier. Defaults to None.
            convert_here (bool, optional): 是否在该类下进行features转换，默认为True。如果已经在Dataset里全部转换完成，这里就可以赋值为False.
        """
        self.config = config
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.convert_method = convert_method
        self.convert_here = convert_here
        
    def __call__(self,batch_data):
        """Datacollator调用，与Dataset的get_item方法配合使用

        Args:
            batch_data (List): 一个batch的data

        Returns:
            model_params: 模型参数的batch形式
        """
        # sourcery skip: assign-if-exp, swap-if-expression
        if self.convert_here:
            features = self.convert_method(self.config, batch_data, self.tokenizer, self.vocab)
        else:
            features = batch_data

        model_params = defaultdict(list)
        for one_item in features:
            for k,v in one_item.items():
                model_params[k].append(v)

        for k in model_params:
            if type(model_params[k][0]) is torch.Tensor:
                model_params[k] = torch.stack(model_params[k])
        return model_params


   
def _torch_collate_batch(examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    import numpy as np
    import torch

    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple, np.ndarray)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    length_of_first = examples[0].size(0)

    # Check if padding is necessary.

    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
        return torch.stack(examples, dim=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(x.size(0) for x in examples)
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0] :] = example
    return result

@dataclass
class MyDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
 
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        
        # Handle dict or lists with proper padding and conversion to tensor.
        batch_encoding = self.tokenizer(examples, add_special_tokens=True, truncation=True, max_length=128)
        examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in batch_encoding['input_ids']]
        if isinstance(examples[0], (dict, BatchEncoding)):
            batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch