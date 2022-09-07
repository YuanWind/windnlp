#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@File    :   base_model.py
@Time    :   2022/07/25 10:21:21
@Author  :   Yuan Wind
@Desc    :   None
'''
import logging
logger = logging.getLogger(__name__.replace('_', ''))
import torch.nn as nn
from scripts.config import MyConfigs




def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result

class BaseModel(nn.Module):
    def __init__(self, config:MyConfigs) -> None:
        super().__init__()
        self.config = config
        self.cur_batch_state = None
        
    
    def optimizer_grouped_parameters(self, weight_decay):
        """为模型参数设置不同的优化器超参，比如分层学习率等，此处默认为hf的初始化方式
        """
        decay_parameters = get_parameter_names(self, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if n in decay_parameters],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
            },
        ]
        return optimizer_grouped_parameters
    
    def forward(self):
        raise NotImplementedError
    
    def calc_loss(self):
        raise NotImplementedError
