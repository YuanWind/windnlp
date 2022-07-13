
nohup python code/main.py --config_file 'code/wind_configs/offline.cfg' --postfix 'baseline' \
				--vocab_file 'data_gaiic_wind/temp_data/vocab.pkl' \
			    > 'logs/nohup.baseline.txt' 2>&1 &

nohup python code/main.py --config_file 'code/wind_configs/offline.cfg' --postfix 'rdrop' \
				--vocab_file 'data_gaiic_wind/temp_data/vocab.pkl' \
				--alpha 2 \
			    > 'logs/nohup.rdrop.txt' 2>&1 &

nohup python code/main.py --config_file 'code/wind_configs/offline.cfg' --postfix 'fgm' \
				--vocab_file 'data_gaiic_wind/temp_data/vocab.pkl' \
				--adversarival_type fgm \
			    > 'logs/nohup.fgm.txt' 2>&1 &

nohup python code/main.py --config_file 'code/wind_configs/offline.cfg' --postfix 'pgd' \
				--vocab_file 'data_gaiic_wind/temp_data/vocab.pkl' \
				--adversarival_type pgd \
			    > 'logs/nohup.pgd.txt' 2>&1 &

nohup python code/main.py --config_file 'code/wind_configs/offline.cfg' --postfix 'ema' \
				--vocab_file 'data_gaiic_wind/temp_data/vocab.pkl' \
				--other_tricks '[ema]' \
			    > 'logs/nohup.ema.txt' 2>&1 &


nohup python code/main.py --config_file 'code/wind_configs/offline.cfg' --postfix 'all_tricks_52' \
				--vocab_file 'data_gaiic_wind/temp_data/vocab.pkl' \
				--other_tricks '[ema]' \
				--adversarival_type fgm \
				--alpha 2 \
				--head_size 52 \
			    > 'logs/nohup.all_tricks_52.txt' 2>&1 &











nohup python code/main.py --config_file 'code/wind_configs/offline.cfg' --postfix 'best_666' \
				--per_device_train_batch_size 16 \
				--num_train_epochs 30 \
				--seed 666 \
				--fp16 true \
				--vocab_file 'data_gaiic_wind/temp_data/vocab.pkl' \
			    > 'logs/nohup.best_666.txt' 2>&1 &

nohup python code/main.py --config_file 'code/wind_configs/offline.cfg' --postfix 'baseline_softlabel' \
				--fp16 false \
				--vocab_file 'data_gaiic_wind/temp_data/vocab.pkl' \
			    > 'logs/nohup.baseline_softlabel.txt' 2>&1 &


nohup python code/main.py --config_file 'code/wind_configs/offline.cfg' --postfix 'baseline_ema_8300_alpha_2.25' \
				--ema_start_steps 8300 \
				--alpha 2.25 \
				--vocab_file 'data_gaiic_wind/temp_data/vocab.pkl' \
			    > 'logs/nohup.baseline_ema_8300_alpha_2.25.txt' 2>&1 &

nohup python code/main.py --config_file 'code/wind_configs/offline.cfg' --postfix 'baseline_alpha2.5' \
				--alpha 2.5 \
				--vocab_file 'data_gaiic_wind/temp_data/vocab.pkl' \
			    > 'logs/nohup.baseline_alpha2.5.txt' 2>&1 &

nohup python code/main.py --config_file 'code/wind_configs/offline.cfg' --postfix 'baseline_lookahead_3' \
				--per_device_train_batch_size 16 \
				--num_train_epochs 30 \
				--other_tricks '[ema, lookahead]' \
				--lookahead_alpha 0.3 \
				--fp16 true \
				--vocab_file 'data_gaiic_wind/temp_data/vocab.pkl' \
			    > 'logs/nohup.baseline_lookahead_3.txt' 2>&1 &

nohup python code/main.py --config_file 'code/wind_configs/offline.cfg' --postfix 'no_ema' \
				--per_device_train_batch_size 16 \
				--other_tricks '[]' \
				--fp16 true \
				--vocab_file 'data_gaiic_wind/temp_data/vocab.pkl' \
			    > 'logs/nohup.no_ema.txt' 2>&1 &

nohup python code/main.py --config_file 'code/wind_configs/offline.cfg' --postfix 'new_nezha_5epoch' \
				--per_device_train_batch_size 16 \
				--fp16 true \
				--vocab_file 'data_gaiic_wind/temp_data/vocab.pkl' \
				--pretrained_model_name_or_path 'data_gaiic_wind/outputs_pretrain_nezha/pretrain_nezha' \
			    > 'logs/nohup.new_nezha_5epoch.txt' 2>&1 &	

nohup python code/main.py --config_file 'code/wind_configs/offline.cfg' --postfix 'bs32' \
				--per_device_train_batch_size 32 \
				--fp16 true \
				--vocab_file 'data_gaiic_wind/temp_data/vocab.pkl' \
			    > 'logs/nohup.bs32.txt' 2>&1 &

nohup python code/main.py --config_file 'code/wind_configs/offline.cfg' --postfix 'seed666' \
				--seed 666 \
				--fp16 true \
				--vocab_file 'data_gaiic_wind/temp_data/vocab.pkl' \
			    > 'logs/nohup.seed666.txt' 2>&1 &











