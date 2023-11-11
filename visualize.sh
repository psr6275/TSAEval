real_path="--real-path ./data/ngsim/ns_test_le.csv"
syn_path="--syn-path ./data/ngsim/ngsim_ns_syn.csv"
config_path="--config-path ./configs/default/ngsim_data_config.json"
save_path="--save-path ./results/ngsim_ns_viz_exp.pkl"
datainfo_path="--datainfo-path ./data/ngsim/data_info.pkl"
python visualize.py $real_path $syn_path $config_path $save_path $datainfo_path
syn_path2="--syn-path ./data/ngsim/ngsim_rtf_syn.csv"
save_path2="--save-path ./results/ngsim_rtf_viz_exp.pkl"
python visualize.py $real_path $syn_path2 $config_path $save_path2 $datainfo_path
syn_path3="--syn-path ./data/ngsim/ngsim_actgan_syn.csv"
save_path3="--save-path ./results/ngsim_actgan_viz_exp.pkl"
python visualize.py $real_path $syn_path3 $config_path $save_path3 $datainfo_path
syn_path4="--syn-path ./data/ngsim/ngsim_ctgan_syn.csv"
save_path4="--save-path ./results/ngsim_ctgan_viz_exp.pkl"
python visualize.py $real_path $syn_path4 $config_path $save_path4 $datainfo_path