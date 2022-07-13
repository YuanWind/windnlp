
nohup python code/main.py --config_file 'code/wind_configs/offline.cfg' --postfix 'fgm_rdrop_ema' \
				--num_train_epochs 8 \
				--adversarival_type fgm \
				--other_tricks '[ema]' \
				--ema_start_steps 8000 \
				--fgm_e 0.3 \
				--alpha 2 \
				--per_device_train_batch_size 16 \
				--fp16 true \
				--vocab_file 'data_gaiic_wind/temp_data/vocab.pkl' \
				--pretrained_model_name_or_path 'data_gaiic_wind/nezha/nezha-cn-base' \
			    > 'logs/nohup.fgm_rdrop_ema.txt' 2>&1 &


nohup python code/main.py --config_file 'code/wind_configs/offline.cfg' --postfix 'fgm_ema' \
				--num_train_epochs 8 \
				--adversarival_type fgm \
				--other_tricks '[ema]' \
				--ema_start_steps 8000 \
				--fgm_e 0.3 \
				--per_device_train_batch_size 16 \
				--fp16 true \
				--vocab_file 'data_gaiic_wind/temp_data/vocab.pkl' \
				--pretrained_model_name_or_path 'data_gaiic_wind/nezha/nezha-cn-base' \
			    > 'logs/nohup.fgm_ema.txt' 2>&1 &

nohup python code/main.py --config_file 'code/wind_configs/offline.cfg' --postfix 'rdrop_ema' \
				--num_train_epochs 8 \
				--other_tricks '[ema]' \
				--ema_start_steps 8000 \
				--alpha 2 \
				--per_device_train_batch_size 16 \
				--fp16 true \
				--vocab_file 'data_gaiic_wind/temp_data/vocab.pkl' \
				--pretrained_model_name_or_path 'data_gaiic_wind/nezha/nezha-cn-base' \
			    > 'logs/nohup.rdrop_ema.txt' 2>&1 &



