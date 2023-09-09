real_path="--real-path ./data/rss/sampled_merged_data1.csv"
syn_path="--syn-path ./data/rss/sampled_merged_data2.csv"
config_path="--config-path ./configs/default/rss_data_config.json"
save_path="--save-path ./results/rss_viz_exp.pkl"
python visualize.py $real_path $syn_path $config_path $save_path