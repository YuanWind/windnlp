
nohup python projects/main.py --config_file projects/main_chinese.cfg --postfix 'ada_05' --model_type 'ori_ada'\
		--ada_temp 1.0 \
		> projects/logs/log_ada_05.txt 2>&1 &

sleep 3

nohup python projects/main.py --config_file projects/main_chinese.cfg --postfix 'ada_06' --model_type 'ori_ada'\
		--ada_temp 0.5 \
		> projects/logs/log_ada_06.txt 2>&1 &

sleep 3

nohup python projects/main.py --config_file projects/main_chinese.cfg --postfix 'ada_07' --model_type 'ori_ada'\
		--ada_temp 2 \
		> projects/logs/log_ada_07.txt 2>&1 &

sleep 3

nohup python projects/main.py --config_file projects/main_chinese.cfg --postfix 'bart_08' --model_type 'ori_bart'\
		--learning_rate 5e-5 \
		> projects/logs/log_bart_08.txt 2>&1 &

sleep 3

nohup python projects/main.py --config_file projects/main_chinese.cfg --postfix 'bart_09' --model_type 'ori_bart'\
		--learning_rate 2e-5 \
		> projects/logs/bart_09.txt 2>&1 &

sleep 3

nohup python projects/main.py --config_file projects/main_chinese.cfg --postfix 'ada_bart_10' --model_type 'ada_bart'\
		--learning_rate 5e-5 \
		> projects/logs/ada_bart_10.txt 2>&1 &

sleep 3


nohup python projects/main.py --config_file projects/main_chinese.cfg --postfix 'ada_bart_11' --model_type 'ada_bart'\
		--learning_rate 2e-5 \
		> projects/logs/ada_bart_11.txt 2>&1 &