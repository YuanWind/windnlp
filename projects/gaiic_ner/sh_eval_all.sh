
best_model_file=$1

python code/eval.py --config_file 'code/wind_configs/submit.cfg' --postfix 'submit' \
        --path_word '/home/mw/input/data_gaiic_wind1418/data_gaiic_wind/temp_data/train_dev/0.90_dev.json'\
        --best_model_file ${best_model_file}

python code/eval.py --config_file 'code/wind_configs/submit.cfg' --postfix 'submit' \
        --path_word '/home/mw/input/data_gaiic_wind1418/data_gaiic_wind/temp_data/train_dev/0.91_dev.json'\
        --best_model_file ${best_model_file}


python code/eval.py --config_file 'code/wind_configs/submit.cfg' --postfix 'submit' \
        --path_word '/home/mw/input/data_gaiic_wind1418/data_gaiic_wind/temp_data/train_dev/0.92_dev.json'\
        --best_model_file ${best_model_file}

python code/eval.py --config_file 'code/wind_configs/submit.cfg' --postfix 'submit' \
        --path_word '/home/mw/input/data_gaiic_wind1418/data_gaiic_wind/temp_data/train_dev/0.93_dev.json'\
        --best_model_file ${best_model_file}

python code/eval.py --config_file 'code/wind_configs/submit.cfg' --postfix 'submit' \
        --path_word '/home/mw/input/data_gaiic_wind1418/data_gaiic_wind/temp_data/train_dev/0.94_dev.json'\
        --best_model_file ${best_model_file}

python code/eval.py --config_file 'code/wind_configs/submit.cfg' --postfix 'submit' \
        --path_word '/home/mw/input/data_gaiic_wind1418/data_gaiic_wind/temp_data/train_dev/0.95_dev.json'\
        --best_model_file ${best_model_file}

python code/eval.py --config_file 'code/wind_configs/submit.cfg' --postfix 'submit' \
        --path_word '/home/mw/input/data_gaiic_wind1418/data_gaiic_wind/temp_data/train_dev/0.96_dev.json'\
        --best_model_file ${best_model_file}

python code/eval.py --config_file 'code/wind_configs/submit.cfg' --postfix 'submit' \
        --path_word '/home/mw/input/data_gaiic_wind1418/data_gaiic_wind/temp_data/train_dev/0.97_dev.json'\
        --best_model_file ${best_model_file}

