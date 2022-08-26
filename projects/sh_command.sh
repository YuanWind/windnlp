nohup python projects/main.py --config_file projects/main_chinese.cfg --postfix '31' --model_type 'ori_bart'\
		--learning_rate 3e-5 --dropout 0.2 --attn_dropout 0.2 --add_triples False --use_copy_mechanism False \
		--num_train_epochs 30 \
		> projects/logs/31.txt 2>&1 &

nohup python projects/main.py --config_file projects/main_chinese.cfg --postfix '32' --model_type 'ori_bart'\
		--learning_rate 3e-5 --dropout 0.2 --attn_dropout 0.2 --add_triples True --use_copy_mechanism False \
		--num_train_epochs 30 \
		> projects/logs/32.txt 2>&1 &

nohup python projects/main.py --config_file projects/main_chinese.cfg --postfix '33' --model_type 'ori_bart'\
		--learning_rate 3e-5 --dropout 0.2 --attn_dropout 0.2 --add_triples True --use_copy_mechanism True \
		--num_train_epochs 30 \
		> projects/logs/33.txt 2>&1 &


nohup python projects/main.py --config_file projects/main_chinese.cfg --postfix '34' --model_type 'ph_bart'\
		--learning_rate 3e-5 --dropout 0.2 --attn_dropout 0.2 --add_triples True --use_copy_mechanism True \
		--num_train_epochs 30 \
		> projects/logs/34.txt 2>&1 &

nohup python projects/main.py  --do_train False --config_file 'projects/outs/34/main_chinese.cfg' \
		> projects/logs/34_eval.txt 2>&1 &

nohup python projects/main.py --config_file projects/main_chinese.cfg --postfix '35' --model_type 'ph_bart'\
		--learning_rate 3e-5 --dropout 0.2 --attn_dropout 0.2 --add_triples False --use_copy_mechanism False \
		--num_train_epochs 30 \
		> projects/logs/35.txt 2>&1 &

nohup python projects/main.py  --do_train False --config_file 'projects/outs/35/main_chinese.cfg' \
		> projects/logs/35_eval.txt 2>&1 &

nohup python projects/main.py --config_file projects/main_chinese.cfg --postfix '36' --model_type 'ph_bart'\
		--learning_rate 3e-5 --dropout 0.2 --attn_dropout 0.2 --add_triples True --use_copy_mechanism False \
		--num_train_epochs 30 \
		> projects/logs/36.txt 2>&1 &

nohup python projects/main.py  --do_train False --config_file 'projects/outs/36/main_chinese.cfg' \
		> projects/logs/36_eval.txt 2>&1 &

nohup python projects/main.py --config_file projects/main_chinese.cfg --postfix '37' --model_type 'ori_bart'\
		--learning_rate 3e-5 --dropout 0.2 --attn_dropout 0.2 --add_triples False --use_copy_mechanism False \
		--num_train_epochs 30 --share_encoder_for_pi False \
		> projects/logs/37.txt 2>&1 &

nohup python projects/main.py --config_file projects/main_chinese.cfg --postfix '38' --model_type 'ph_bart'\
		--learning_rate 3e-5 --dropout 0.2 --attn_dropout 0.2 --add_triples True --use_copy_mechanism False \
		--num_train_epochs 10 --share_encoder_for_pi False \
		> projects/logs/38.txt 2>&1 &




nohup python projects/main.py --config_file projects/main_chinese.cfg --postfix 'test' --model_type 'ph_bart'\
		--learning_rate 3e-5 --dropout 0.2 --attn_dropout 0.2 --add_triples True --use_copy_mechanism False \
		--num_train_epochs 30 \
		> projects/logs/test.txt 2>&1 &
		
nohup deepspeed projects/main.py --config_file projects/main_chinese.cfg --postfix 'test1' --model_type 'ph_bart'\
		--learning_rate 3e-5 --dropout 0.2 --attn_dropout 0.2 --add_triples True --use_copy_mechanism False \
		--num_train_epochs 30  --deepspeed projects/deepspeed.json \
		> projects/logs/test1.txt 2>&1 &











nohup python projects/main.py --config_file projects/main_chinese.cfg >projects/log_chinese.txt 2>&1 &




nohup python projects/main.py --config_file projects/main.cfg --postfix 'daily_100' \
		> projects/log_daily_100.txt 2>&1 &


# nohup python projects/main.py --config_file projects/main.cfg --postfix 'ori_ada' --model_type 'ori_ada'\
# 		--learning_rate 2e-4 \
# 		> projects/log_ori_ada.txt 2>&1 &


nohup python projects/main.py --config_file projects/main.cfg --postfix 'bart_constant' --model_type 'ori_bart' \
		--learning_rate 1e-4 \
		> projects/log_ori_bart_constant.txt 2>&1 &

nohup python projects/main.py --config_file projects/main.cfg --postfix 'ada_bart' --model_type 'ada_bart' \
		--learning_rate 1e-4 \
		> projects/log_ada_bart_constant.txt 2>&1 &

# 待执行
nohup python projects/main.py --config_file projects/main.cfg --postfix 'ada_constant_lr2' --model_type 'ori_ada' \
		--learning_rate 2e-4 \
		> projects/log_ori_ada_constant_lr2.txt 2>&1 &



# chinese
nohup python projects/main.py --config_file projects/main_chinese.cfg --postfix 'ada_bs_16' --model_type 'ori_ada'\
		> projects/logs/log_ada_bs_16.txt 2>&1 &

nohup python projects/main.py --config_file projects/main_chinese.cfg --postfix 'bart_lr_3_4' --model_type 'ori_bart'\
		--learning_rate 2e-3 \
		> projects/logs/log_bart_lr_3_4.txt 2>&1 &
		
nohup python projects/main.py --config_file projects/main_chinese.cfg --postfix 'ada_bart_lr_3_4' --model_type 'ada_bart'\
		--learning_rate 2e-3 \
		> projects/logs/log_ada_bart_lr_3_4.txt 2>&1 &

# eval
python projects/main.py --do_train false --config_file projects/main.cfg \
		--postfix 'ori_ada' --model_type 'ori_ada' \
		--best_model_file 'projects/outs/ori_ada/checkpoint-32900/pytorch_model.bin'

python projects/main.py --do_train false --config_file projects/outs/bart_20/main_chinese.cfg \
		--postfix 'only_eval' --model_type 'ori_bart' \
		--best_model_file 'projects/outs/bart_20/best_model/best.pt'

python projects/main.py --do_train false --config_file projects/main_chinese.cfg --postfix 'ori_ada_zh' \
							--model_type 'ori_ada' \
							--best_model_file 'projects/outs/ori_ada_zh/best_model/best.pt'

python projects/main.py --do_train false --config_file projects/main_chinese.cfg --postfix 'ada_bart_zh' \
							--model_type 'ada_bart' \
							--best_model_file 'projects/outs/ada_bart_zh/best_model/best.pt'


nohup python projects/main.py --config_file projects/main_chinese.cfg --postfix 'bart_15' --model_type 'ori_bart'\
		--learning_rate 2e-5 --dropout 0.1 --attn_dropout 0.1 --add_triples False \
		> projects/logs/bart_15.txt 2>&1 &

nohup python projects/main.py --config_file projects/main_chinese.cfg --postfix '16_eval' --model_type 'ori_bart'\
		--learning_rate 2e-5 --dropout 0.1 --attn_dropout 0.1 --add_triples True \
		> projects/logs/16_eval.txt 2>&1 &

nohup python projects/main.py --config_file projects/main_chinese.cfg --postfix 'bart_17' --model_type 'ori_bart'\
		--learning_rate 2e-5 --dropout 0.1 --attn_dropout 0.1 --add_triples True --num_train_epochs 10 \
		> projects/logs/bart_17.txt 2>&1 &



nohup python projects/main.py --config_file projects/main_chinese.cfg --postfix 'ada_bart_18' --model_type 'ada_bart'\
		--learning_rate 2e-5 --dropout 0.1 --attn_dropout 0.1 --add_triples True \
		> projects/logs/ada_bart_18.txt 2>&1 &

nohup python projects/main.py --config_file projects/main_chinese.cfg --postfix 'ada_bart_19' --model_type 'ada_bart'\
		--learning_rate 2e-5 --dropout 0.1 --attn_dropout 0.1 --add_triples False \
		> projects/logs/ada_bart_19.txt 2>&1 &


nohup python projects/main.py --config_file projects/main_chinese.cfg --postfix 'bart_20' --model_type 'ori_bart'\
		--learning_rate 2e-5 --dropout 0.1 --attn_dropout 0.1 --add_triples True --num_train_epochs 15 \
		> projects/logs/bart_20.txt 2>&1 &


nohup python projects/main.py --config_file projects/main_chinese.cfg --postfix 'bart_21' --model_type 'ori_bart'\
		--learning_rate 2e-5 --dropout 0.1 --attn_dropout 0.1 --add_triples False --num_train_epochs 15 \
		> projects/logs/bart_21.txt 2>&1 &

nohup python projects/main.py --config_file projects/main_chinese.cfg --postfix '23' --model_type 'ori_bart'\
		--learning_rate 2e-5 --dropout 0.1 --attn_dropout 0.1 --add_triples False --num_train_epochs 5 \
		> projects/logs/23.txt 2>&1 &

nohup python projects/main.py --config_file projects/main_chinese.cfg --postfix 'only_eval' --model_type 'ori_bart'\
		--learning_rate 2e-5 --dropout 0.1 --attn_dropout 0.1 --add_triples False --num_train_epochs 5 \
		--do_train false --best_model_file 'projects/outs/bart_22/best_model/epoch2.pt' \
		--pretrained_model_name_or_path 'fnlp/bart-large-chinese' > projects/outs/bart_22/epoch2_pred.txt 2>&1 &


nohup python projects/main.py --config_file projects/main_chinese.cfg --postfix '24' --model_type 'ori_bart'\
		--learning_rate 2e-5 --dropout 0.1 --attn_dropout 0.1 --add_triples False --num_train_epochs 10 \
		> projects/logs/24.txt 2>&1 &

nohup python projects/main.py --config_file projects/main_chinese.cfg --postfix '26' --model_type 'ori_bart'\
		--learning_rate 2e-5 --dropout 0.1 --attn_dropout 0.1 --add_triples True --num_train_epochs 10 \
		> projects/logs/26.txt 2>&1 &

nohup python projects/main.py --config_file projects/main_chinese.cfg --postfix 'eval' --model_type 'ori_bart'\
		--learning_rate 3e-5 --dropout 0.1 --attn_dropout 0.1 --add_triples True --num_train_epochs 30 \
		--do_train false --best_model_file 'projects/outs/27/best_model/epoch9.pt' \
		> projects/logs/27_epoch8_eval.txt 2>&1 &

nohup python projects/main.py --config_file projects/main_chinese.cfg --postfix '28' --model_type 'ori_bart'\
		--learning_rate 3e-5 --dropout 0.1 --attn_dropout 0.1 --add_triples True --num_train_epochs 30 \
		> projects/logs/28.txt 2>&1 &

nohup python projects/main.py --config_file projects/main_chinese.cfg --postfix '29' --model_type 'ori_bart'\
		--learning_rate 3e-5 --dropout 0.1 --attn_dropout 0.1 --add_triples True --num_train_epochs 20 \
		> projects/logs/29.txt 2>&1 &

nohup python projects/main.py --config_file projects/main_chinese.cfg --postfix '30' --model_type 'ori_bart'\
		--learning_rate 3e-5 --dropout 0.1 --attn_dropout 0.1 --add_triples True --num_train_epochs 20 --do_train false \
		> projects/logs/30.txt 2>&1 &