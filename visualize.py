import os
import re
import copy
import warnings
import pandas as pd

from config_io import Config
from sdmetrics.reports.timeseries import QualityReport

from netshare.configs import default as default_configs
from eval_utils import read_file

# class RTFGenerator(object):
#     def __init__(self, config):
#         self.parent = 1
#         self.child
    
#     def generate(self,):
#         dd = 0
    
#     def _get_visualization_folder(self, work_folder):
#         return os.path.join(work_folder, "visulization")
    
#     # def visualize(self, work_folder, real_path, syn_path):
#         # os.makedirs(self._get_visualization_folder(work_folder), exist_ok=True)
def create_sdmetrics_config_rtf(
        config_path,
        comparison_type='both'
):  
    sdmetrics_config = {
        "metadata": {
            "fields": {}
        },
        "config": {
            "metrics": {
                "fidelity": []
            }
        }
    }

    config = Config.load_from_file(config_path, 
                                   default_search_paths = default_configs.__path__)
    pre_post_processor_config = Config(config["global_config"])
    pre_post_processor_config.update(
            config['pre_post_processor']['config'])
    
    for i, field in enumerate(pre_post_processor_config.metadata +
                              pre_post_processor_config.timeseries):
        if field in pre_post_processor_config.metadata:
            metric_class_name = "Single attribute distributional similarity"
            class_name = "AttrDistSimilarity"
        elif field in pre_post_processor_config.timeseries:
            metric_class_name = "Single feature distributional similarity"
            class_name = "FeatureDistSimilarity"

        if 'bit' in getattr(field, 'encoding', '') or \
            'word2vec' in getattr(field, 'encoding', '') or \
                'categorical' in getattr(field, 'encoding', ''):
            sdmetrics_config["metadata"]["fields"][
                field.column] = {
                "type": "categorical"}
        if getattr(field, 'type', '') == 'float':
            sdmetrics_config["metadata"]["fields"][
                field.column] = {
                "type": "numerical"}
        sdmetrics_config["config"]["metrics"]["fidelity"].append(
            {
                metric_class_name: {
                    "class": class_name,
                    "target_list": [[field.column]],
                    "configs": {
                        "categorical_mapping": getattr(field, 'categorical_mapping', True),
                        "comparison_type": comparison_type
                    }
                }
            }
        )    
    sdmetrics_config["metadata"]["fields"][
                "index"] = {
                "type": "integer"}
    sdmetrics_config["config"]["metrics"]["fidelity"].append(
            {
                "Session length distributional similarity": {
                    "class": "SessionLengthDistSimilarity",
                    "configs": {
                        "comparison_type": comparison_type
                    }
                }
            }
        )
    if pre_post_processor_config.timestamp.generation:
        sdmetrics_config["metadata"]["fields"][
            pre_post_processor_config.timestamp.column] = {
            "type": "numerical"}
        sdmetrics_config["config"]["metrics"]["fidelity"].append(
            {
                "Single feature distributional similarity": {
                    "class": "FeatureDistSimilarity",
                    "target_list": [
                        [
                            pre_post_processor_config.timestamp.column
                        ]
                    ],
                    "configs": {
                        "comparison_type": comparison_type
                    }
                }
            }
        )
    sdmetrics_config["metadata"]["entity_columns"] = ['index']
    # sdmetrics_config["metadata"]["entity_columns"] = [
    #     field.column for field in pre_post_processor_config.metadata
    # ]
    sdmetrics_config["metadata"]["sequence_index"] = pre_post_processor_config.timestamp.column if pre_post_processor_config.timestamp.generation else None
    sdmetrics_config["metadata"]["context_columns"] = [
        field.column for field in pre_post_processor_config.metadata
    ]

    return sdmetrics_config



def visualize(real_df, synth_df, config_path, method='rtf'):

    if method == 'rtf':
        sdmetrics_config= create_sdmetrics_config_rtf(config_path, 'both')
    else:
        from netshare.pre_post_processors.netshare.util import create_sdmetrics_config
        config = Config.load_from_file(config_path, 
                                   default_search_paths = default_configs.__path__)
        pre_post_processor_config = Config(config["global_config"])
        pre_post_processor_config.update(
                        config['pre_post_processor']['config'])
        sdmetrics_config = create_sdmetrics_config(pre_post_processor_config, 'both')

    my_report = QualityReport(
            config_dict=sdmetrics_config['config'])
    print("total cols:",synth_df.columns)
    # inc_cols = list(set(synth_df.columns) - set(['index']))
    inc_cols = synth_df.columns
    # print("include cols: ",inc_cols)
    my_report.generate(real_df[inc_cols], synth_df[inc_cols],
                        sdmetrics_config['metadata'])
    my_report.visualize()


if __name__ =='__main__':
    real_path = './data/ns_train_df.csv'
    # real_path = './data/ns_test_df.parquet'
    # real_path = './data/ns_train_10000_df.csv'
    # syn_path = './results/rtf/230811/merged_synthetic_data.csv'
    # syn_path = './results/rtf/merged_synthetic_data.csv'
    syn_path = './results/netshare/230808_flowid/merged_synth_df.csv'
    # syn_path = './results/netshare/230816/merged_synth_df.csv'
    config_path = './netshare/configs/config_conviva_session_nodp.json'
    # config_path = './netshare/configs/config_conviva_session_nodp_tot.json'
    
    method = 'netshare'

    assert os.path.isfile(real_path) and os.path.isfile(syn_path)
    dtype = {'state':str, 'isp':str}
    real_df = read_file(real_path)
    syn_df = pd.read_csv(syn_path)
    for dt in dtype:
        real_df[dt] = real_df[dt].astype(dtype[dt])
        syn_df[dt] = syn_df[dt].astype(dtype[dt])
    # syn_df = read_file(syn_path, dtype)

    print("Start visualize ====>")
    visualize(real_df, syn_df, config_path, method)
