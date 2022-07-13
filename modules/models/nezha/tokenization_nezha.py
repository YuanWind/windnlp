# -*- encoding: utf-8 -*-
'''
@File    :   tokenization_nezha.py
@Time    :   2022/04/18 09:02:51
@Author  :   Yuan Wind
@Desc    :   None
'''
import logging
logger = logging.getLogger(__name__.replace('_', ''))
from transformers.models.bert import BertTokenizer, BertTokenizerFast

class NeZhaTokenizer(BertTokenizer):
	def __init__(self, vocab_file, do_lower_case=True, do_basic_tokenize=True, never_split=None, unk_token="[UNK]", sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]", mask_token="[MASK]", tokenize_chinese_chars=True, strip_accents=None, **kwargs):
		super().__init__(vocab_file, do_lower_case, do_basic_tokenize, never_split, unk_token, sep_token, pad_token, cls_token, mask_token, tokenize_chinese_chars, strip_accents, **kwargs)


class NeZhaTokenizerFast(BertTokenizerFast):
	def __init__(self, vocab_file=None, tokenizer_file=None, do_lower_case=True, unk_token="[UNK]", sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]", mask_token="[MASK]", tokenize_chinese_chars=True, strip_accents=None, **kwargs):
		super().__init__(vocab_file, tokenizer_file, do_lower_case, unk_token, sep_token, pad_token, cls_token, mask_token, tokenize_chinese_chars, strip_accents, **kwargs)
		