python code/main.py --config_file 'data_gaiic_wind/output_dir/output_baseline_ema_8500_alpha_2.25/offline.cfg' \
			   --do_train false --fp16 false --threshold 0


python main.py --config_file 'data_gaiic_wind/output_dir/output_no_rdrop/baseline.cfg' \
			   --dev_pred_file 'data_gaiic_wind/predict/dev_pred_no_rdrop.json' \
			   --test_pred_file 'data_gaiic_wind/predict/test_pred_no_rdrop.json' \
			   --do_train false --fp16 false