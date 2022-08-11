export CUDA_VISIBLE_DEVICE=0
nohup python projects/main.py --config_file projects/main_chinese.cfg --postfix 'ada_05' --model_type 'ori_ada'\
		--ada_temp 1.0 --resume_from_checkpoint 'projects/outs/ada_05/checkpoint-49990'\
		> projects/logs/log_ada_05.txt 2>&1 &