
sleep 3

nohup python projects/main.py --config_file projects/main_chinese.cfg --postfix 'ada_06' --model_type 'ori_ada'\
		--ada_temp 0.5  --resume_from_checkpoint 'projects/outs/ada_06/checkpoint-49990' \
		> projects/logs/log_ada_06.txt 2>&1 &

sleep 3

nohup python projects/main.py --config_file projects/main_chinese.cfg --postfix 'ada_07' --model_type 'ori_ada'\
		--ada_temp 2 --resume_from_checkpoint 'projects/outs/ada_07/checkpoint-54989' \
		> projects/logs/log_ada_07.txt 2>&1 &

sleep 3

nohup python projects/main.py --config_file projects/main_chinese.cfg --postfix 'bart_08' --model_type 'ori_bart'\
		--learning_rate 5e-5 --resume_from_checkpoint 'projects/outs/bart_08/checkpoint-69986' \
		> projects/logs/log_bart_08.txt 2>&1 &

sleep 3

nohup python projects/main.py --config_file projects/main_chinese.cfg --postfix 'bart_12' --model_type 'ori_bart'\
		--learning_rate 2e-5 --dropout 0.1 --attn_dropout 0.1 \
		> projects/logs/bart_12.txt 2>&1 &

nohup python projects/main.py --config_file projects/main_chinese.cfg --postfix 'bart_13' --model_type 'ori_bart'\
		--learning_rate 2e-5 --dropout 0.1 --attn_dropout 0.1 --pretrained_model_name_or_path 'facebook/bart-base' \
		> projects/logs/bart_13.txt 2>&1 &

sleep 3

nohup python projects/main.py --config_file projects/main_chinese.cfg --postfix 'ada_bart_10' --model_type 'ada_bart'\
		--learning_rate 5e-5 --resume_from_checkpoint 'projects/outs/ada_bart_10/checkpoint-44991' \
		> projects/logs/ada_bart_10.txt 2>&1 &

# sleep 3




# nohup python projects/main.py --config_file projects/main_chinese.cfg --postfix 'ada_bart_11' --model_type 'ada_bart'\
# 		--learning_rate 2e-5 \
# 		> projects/logs/ada_bart_11.txt 2>&1 &