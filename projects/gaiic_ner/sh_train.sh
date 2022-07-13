export CUDA_VISIBLE_DEVICES=0
nohup python code/main.py --config_file 'code/wind_configs/offline.cfg' --postfix 'fold_0' \
				--adversarival_type fgm \
				--num_train_epochs 11 \
				--fgm_e 0.3 \
				--weight_decay 1e-4 \
				--mtl_bio_w 0.2 \
				--alpha 2 \
				--add_type_emb true \
				--type_w 0.6 \
				--per_device_train_batch_size 16 \
				--fp16 true \
				--vocab_file 'data_gaiic_wind/temp_data/vocabs/vocab_0.pkl' \
			    > 'run_logs/fold_0.log' 2>&1 &

nohup python code/main.py --config_file 'code/wind_configs/offline.cfg' --postfix 'fold_1' \
				--adversarival_type fgm \
				--num_train_epochs 11 \
				--fgm_e 0.3 \
				--weight_decay 1e-4 \
				--mtl_bio_w 0.2 \
				--alpha 2 \
				--add_type_emb true \
				--type_w 0.6 \
				--per_device_train_batch_size 16 \
				--fp16 true \
				--vocab_file 'data_gaiic_wind/temp_data/vocabs/vocab_1.pkl' \
			    > 'run_logs/fold_1.log' 2>&1 &

export CUDA_VISIBLE_DEVICES=1
nohup python code/main.py --config_file 'code/wind_configs/offline.cfg' --postfix 'fold_2' \
				--adversarival_type fgm \
				--num_train_epochs 11 \
				--fgm_e 0.3 \
				--weight_decay 1e-4 \
				--mtl_bio_w 0.2 \
				--alpha 2 \
				--add_type_emb true \
				--type_w 0.6 \
				--per_device_train_batch_size 16 \
				--fp16 true \
				--vocab_file 'data_gaiic_wind/temp_data/vocabs/vocab_2.pkl' \
			    > 'run_logs/fold_2.log' 2>&1 &

nohup python code/main.py --config_file 'code/wind_configs/offline.cfg' --postfix 'fold_3' \
				--adversarival_type fgm \
				--num_train_epochs 11 \
				--fgm_e 0.3 \
				--weight_decay 1e-4 \
				--mtl_bio_w 0.2 \
				--alpha 2 \
				--add_type_emb true \
				--type_w 0.6 \
				--per_device_train_batch_size 16 \
				--fp16 true \
				--vocab_file 'data_gaiic_wind/temp_data/vocabs/vocab_3.pkl' \
			    > 'run_logs/fold_3.log' 2>&1 &

export CUDA_VISIBLE_DEVICES=2
nohup python code/main.py --config_file 'code/wind_configs/offline.cfg' --postfix 'fold_4' \
				--adversarival_type fgm \
				--num_train_epochs 11 \
				--fgm_e 0.3 \
				--weight_decay 1e-4 \
				--mtl_bio_w 0.2 \
				--alpha 2 \
				--add_type_emb true \
				--type_w 0.6 \
				--per_device_train_batch_size 16 \
				--fp16 true \
				--vocab_file 'data_gaiic_wind/temp_data/vocabs/vocab_4.pkl' \
			    > 'run_logs/fold_4.log' 2>&1 &

nohup python code/main.py --config_file 'code/wind_configs/offline.cfg' --postfix 'fold_5' \
				--adversarival_type fgm \
				--num_train_epochs 11 \
				--fgm_e 0.3 \
				--weight_decay 1e-4 \
				--mtl_bio_w 0.2 \
				--alpha 2 \
				--add_type_emb true \
				--type_w 0.6 \
				--per_device_train_batch_size 16 \
				--fp16 true \
				--vocab_file 'data_gaiic_wind/temp_data/vocabs/vocab_5.pkl' \
			    > 'run_logs/fold_5.log' 2>&1 &


export CUDA_VISIBLE_DEVICES=6
nohup python code/main.py --config_file 'code/wind_configs/offline.cfg' --postfix 'fold_6' \
				--adversarival_type fgm \
				--num_train_epochs 11 \
				--fgm_e 0.3 \
				--weight_decay 1e-4 \
				--mtl_bio_w 0.2 \
				--alpha 2 \
				--add_type_emb true \
				--type_w 0.6 \
				--per_device_train_batch_size 16 \
				--fp16 true \
				--vocab_file 'data_gaiic_wind/temp_data/vocabs/vocab_6.pkl' \
			    > 'run_logs/fold_6.log' 2>&1 &

nohup python code/main.py --config_file 'code/wind_configs/offline.cfg' --postfix 'fold_7' \
				--adversarival_type fgm \
				--num_train_epochs 11 \
				--fgm_e 0.3 \
				--weight_decay 1e-4 \
				--mtl_bio_w 0.2 \
				--alpha 2 \
				--add_type_emb true \
				--type_w 0.6 \
				--per_device_train_batch_size 16 \
				--fp16 true \
				--vocab_file 'data_gaiic_wind/temp_data/vocabs/vocab_7.pkl' \
			    > 'run_logs/fold_7.log' 2>&1 &

export CUDA_VISIBLE_DEVICES=7
nohup python code/main.py --config_file 'code/wind_configs/offline.cfg' --postfix 'fold_8' \
				--adversarival_type fgm \
				--num_train_epochs 11 \
				--fgm_e 0.3 \
				--weight_decay 1e-4 \
				--mtl_bio_w 0.2 \
				--alpha 2 \
				--add_type_emb true \
				--type_w 0.6 \
				--per_device_train_batch_size 16 \
				--fp16 true \
				--vocab_file 'data_gaiic_wind/temp_data/vocabs/vocab_8.pkl' \
			    > 'run_logs/fold_8.log' 2>&1 &

nohup python code/main.py --config_file 'code/wind_configs/offline.cfg' --postfix 'fold_9' \
				--adversarival_type fgm \
				--num_train_epochs 11 \
				--fgm_e 0.3 \
				--weight_decay 1e-4 \
				--mtl_bio_w 0.2 \
				--alpha 2 \
				--add_type_emb true \
				--type_w 0.6 \
				--per_device_train_batch_size 16 \
				--fp16 true \
				--vocab_file 'data_gaiic_wind/temp_data/vocabs/vocab_9.pkl' \
			    > 'run_logs/fold_9.log' 2>&1 &