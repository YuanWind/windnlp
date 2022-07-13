import sys
sys.path.extend(['../../','./','../','code'])

from transformers import  TrainingArguments
from wind_data_modules.Datasets import MyLineByLineTextDataset
from wind_data_modules.DataCollator import MyDataCollatorForLanguageModeling
from wind_modules.models.nezha.modeling_nezha import NeZhaForMaskedLM
from wind_modules.models.nezha.configuration_nezha import NeZhaConfig
from transformers import (
    Trainer,
    TrainingArguments,
    set_seed,
)
from wind_modules.models.nezha.tokenization_nezha import NeZhaTokenizerFast

set_seed(777)
## 加载tokenizer和模型
model_path='data_gaiic_wind/nezha/nezha-cn-base'
data_path = 'data_gaiic_wind/train_data/unlabel_data_102w.txt'
save_path = 'data_gaiic_wind/outputs_pretrain_nezha/pretrain_nezha_5'
pretrain_batch_size=128
num_train_epochs=5

tokenizer =  NeZhaTokenizerFast.from_pretrained(model_path)
config=NeZhaConfig.from_pretrained(model_path)
model=NeZhaForMaskedLM.from_pretrained(model_path, config=config)

print('Start to build dataset...')
train_dataset=MyLineByLineTextDataset(file_path=data_path) 
print('Start to build MLM模型的数据DataCollator...')
# MLM模型的数据DataCollator
data_collator = MyDataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
# 训练参数
training_args = TrainingArguments(
    output_dir='data_gaiic_wind/outputs_pretrain_nezha', 
    overwrite_output_dir=True, 
    num_train_epochs=num_train_epochs, 
    learning_rate=4e-5,
    save_strategy = 'epoch',
    per_device_train_batch_size=pretrain_batch_size,
    report_to = ['tensorboard'],
    save_total_limit=1)
# 通过Trainer接口训练模型
trainer = Trainer(
    model=model, args=training_args, data_collator=data_collator, train_dataset=train_dataset)

print('Start to train...')
# 开始训练
trainer.train()
trainer.save_model(save_path)

print('Finish train...')