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
nohup python projects/main.py --config_file projects/main_chinese.cfg --postfix 'ada_base' --model_type 'ori_ada'\
		> projects/logs/log_ada_base.txt 2>&1 &

nohup python projects/main.py --config_file projects/main_chinese.cfg --postfix 'bart_base' --model_type 'ori_bart'\
		> projects/logs/log_bart_base.txt 2>&1 &
		
nohup python projects/main.py --config_file projects/main_chinese.cfg --postfix 'ada_bart_zh' --model_type 'ada_bart'\
		--learning_rate 2e-4 \
		> projects/log_ada_bart_zh.txt 2>&1 &

# eval
python projects/main.py --do_train false --config_file projects/main.cfg \
		--postfix 'ori_ada' --model_type 'ori_ada' \
		--best_model_file 'projects/outs/ori_ada/checkpoint-32900/pytorch_model.bin'

python projects/main.py --do_train false --config_file projects/main.cfg \
		--postfix 'ori_bart' --model_type 'ori_bart' \
		--best_model_file 'projects/outs/bart_constant/checkpoint-32900/pytorch_model.bin'

python projects/main.py --do_train false --config_file projects/main_chinese.cfg --postfix 'ori_ada_zh' \
							--model_type 'ori_ada' \
							--best_model_file 'projects/outs/ori_ada_zh/best_model/best.pt'

python projects/main.py --do_train false --config_file projects/main_chinese.cfg --postfix 'ada_bart_zh' \
							--model_type 'ada_bart' \
							--best_model_file 'projects/outs/ada_bart_zh/best_model/best.pt'


