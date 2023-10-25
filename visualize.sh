real_path="--real-path ./data/ngsim/ns_test_le.csv"
syn_path="--syn-path ./data/ngsim/ngsim_rtf_syn.csv"
config_path="--config-path ./configs/default/ngsim_data_config.json"
save_path="--save-path ./results/ngsim_ns_viz_exp.pkl"
datainfo_path="--datainfo-path ./data/ngsim/data_info.pkl"
python visualize.py $real_path $syn_path $config_path $save_path $datainfo_path