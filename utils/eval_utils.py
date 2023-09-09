import matplotlib.pyplot as plt
import pandas as pd

from config_io import Config

from sdmetrics.reports.timeseries import QualityReport


def read_file(file_path, dtype=None):
    try: 
        df = pd.read_csv(file_path, dtype=dtype)
    except:
        df = pd.read_parquet(file_path)
    return df    

def draw_plots(dfs, skip_cols, num_cols, cat_cols, names = None):
    """
    df1 = real
    df2 = samples
    """
    if names is None:
        names = ["df_%s"%i for i in range(len(dfs))]
    nums = [len(df) for df in dfs]
    
    print("data nums for dfs:", nums)
    for cn in dfs[0].columns:
        if cn in skip_cols:
            print("we skip %s"%cn)
        elif cn in num_cols:
            plt.title("Histogram plot for %s"%cn)
            minx = dfs[0][cn].min()
            maxx = dfs[0][cn].max()
            for df in enumerate(dfs):
                minx = min(minx, df[cn].min())
                maxx = max(maxx, df[cn].max())
            for i, df in enumerate(dfs):
                plt.hist(df[cn], density=True, range=(minx, maxx), alpha=0.5, label=names[i], bins=100)
            plt.legend()
            plt.show()
        elif cn in cat_cols:            
            vcs = []
            for i, df in enumerate(dfs):
                vc = df[cn].astype(str).value_counts()
                if i==0:
                    vk = vc.keys()
                    vcs.append(vc)
                    del vc
                else:
                    iks = list(set(vc.keys())&set(vk))
                    vcs.append(vc[iks])
            plt.title("Bar plot for %s"%cn)
            for i, vc in enumerate(vcs):
                plt.bar(vc.index, vc.values/nums[i], alpha = 0.5,label = names[i])
            plt.legend()
            # plt.title("Bar plot samples for %s"%cn)
            plt.show()
        else:
            print("we skip %s"%cn)

def create_sdmetrics_config(
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

    config = Config.load_from_file(config_path)

    for i, field in enumerate(config.metadata+config.timeseries):
        if field in config.metadata:
            metric_class_name = "Single attribute distributional similarity"
            class_name = "AttrDistSimilarity"
        elif field in config.timeseries:
            metric_class_name = "Single feature distributional similarity"
            class_name = "FeatureDistSimilarity"
        
        if 'bit' in getattr(field, 'encoding', '') or \
            'word2vec' in getattr(field, 'encoding', '') or \
                'categorical' in getattr(field, 'encoding', ''):
            
            sdmetrics_config["metadata"]["fields"][
                field.column] = {
                "type": "categorical"}
            
        if getattr(field, 'type', '') == 'float' or\
            'numerical' in getattr(field, 'encoding', ''):
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
    sdmetrics_config["metadata"]["fields"][config.flowid.column] = {
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
    if config.timestamp.generation:
        sdmetrics_config["metadata"]["fields"][config.timestamp.column]={
            "type":"numerical"
        }
        sdmetrics_config["config"]["metrics"]["fidelity"].append(
            {
                "Single feature distributional similarity": {
                    "class": "FeatureDistSimilarity",
                    "target_list": [
                        [
                            config.timestamp.column
                        ]
                    ],
                    "configs": {
                        "comparison_type": comparison_type
                    }
                }
            }
        )
    sdmetrics_config["metadata"]["entity_columns"] = [config.flowid.column]
    sdmetrics_config["metadata"]["sequence_index"] = config.timestamp.column if config.timestamp.generation else None
    sdmetrics_config["metadata"]["context_columns"] = [
        field.column for field in config.metadata
    ]
    return sdmetrics_config

def viz_sdmetric(real_df, synth_df, config_path, save_path):
    sdmetrics_config = create_sdmetrics_config(config_path,'both')
    my_report = QualityReport(
            config_dict=sdmetrics_config['config'])
    print("total cols:",synth_df.columns)
    # inc_cols = list(set(synth_df.columns) - set(['index']))
    inc_cols = synth_df.columns
    # print("include cols: ",inc_cols)
    my_report.generate(real_df[inc_cols], synth_df[inc_cols],
                        sdmetrics_config['metadata'])
    my_report.visualize()
    my_report.save(filepath=save_path)