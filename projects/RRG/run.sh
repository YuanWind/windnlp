nohup python projects/RRG/main.py --config_file projects/RRG/main_chinese.cfg >projects/RRG/log_chinese.txt 2>&1 &




nohup python projects/RRG/main.py --config_file projects/RRG/main.cfg --postfix 'daily_100' \
		> projects/RRG/log_daily_100.txt 2>&1 &


nohup python projects/RRG/main.py --config_file projects/RRG/main.cfg --postfix 'ori_ada' --model_type 'ori_ada'\
		--learning_rate 2e-4 \
		> projects/RRG/log_ori_ada.txt 2>&1 &




nohup python projects/RRG/main.py --config_file projects/RRG/main.cfg --postfix 'ori_bart' --model_type 'ori_bart' \
		--learning_rate 1e-4 \
		> projects/RRG/log_ori_bart.txt 2>&1 &

nohup python projects/RRG/main.py --config_file projects/RRG/main.cfg --postfix 'ada_bart' --model_type 'ada_bart' \
		--learning_rate 1e-4 \
		> projects/RRG/log_ada_bart.txt 2>&1 &

nohup python projects/RRG/main.py --config_file projects/RRG/main.cfg --postfix 'ori_bart1' --model_type 'ori_bart' \
		--learning_rate 2e-4 \
		> projects/RRG/log_ori_bart1.txt 2>&1 &

nohup python projects/RRG/main.py --config_file projects/RRG/main.cfg --postfix 'ori_bart_25' --model_type 'ori_bart' \
		--learning_rate 2e-4 \
		> projects/RRG/log_ori_bart_25.txt 2>&1 &

# chinese
nohup python projects/RRG/main.py --config_file projects/RRG/main_chinese.cfg --postfix 'ori_ada_zh' --model_type 'ori_ada'\
		--learning_rate 3e-4 \
		> projects/RRG/log_ori_ada_zh.txt 2>&1 &

# eval
python projects/RRG/main.py --do_train false --config_file projects/RRG/main.cfg \
		--postfix 'ori_ada' --model_type 'ori_ada' \
		--best_model_file 'projects/RRG/outs/ori_ada/best_model/best.pt'

python projects/RRG/main.py --do_train false --config_file projects/RRG/main.cfg \
		--postfix 'ori_bart' --model_type 'ori_bart' \
		--best_model_file 'projects/RRG/outs/ori_bart/best_model/best.pt'






