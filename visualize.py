import os
import pandas as pd
import os, argparse

from utils import viz_sdmetric

def main(args):
    real_path = args.real_path
    syn_path  = args.syn_path
    config_path = args.config_path

    assert os.path.isfile(real_path) and os.path.isfile(syn_path), "check the file %s and %s"%(real_path, syn_path)
    
    real_df = pd.read_csv(real_path)
    syn_df = pd.read_csv(syn_path)
    
    assert all(real_df.columns == syn_df.columns), "two datasets have differenc columns!"

    print("Start visualize ====>")
    viz_sdmetric(real_df, syn_df, config_path, args.save_path)

if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='Visualize Using SDMetric_Timeseries')
    parser.add_argument('--real-path', type=str, help='path for real dataset (csv)') 
    parser.add_argument('--syn-path', type=str, help='path for synthetic dataset (csv)') 
    parser.add_argument('--config-path', type=str, help='configuration file for the corresponding dataset') 
    parser.add_argument('--save-path', type=str, help='path to save the visualization results') 
    
    args = parser.parse_args()
    
    main(args)
