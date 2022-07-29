from transformers import BertTokenizer, AutoModel

t = BertTokenizer.from_pretrained('fnlp/bart-base-chinese')
m = AutoModel.from_pretrained('fnlp/bart-base-chinese')