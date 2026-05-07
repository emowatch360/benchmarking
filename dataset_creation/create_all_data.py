import os
from data_config import *
from emowatch_utils import *

labels = ['arousal_current', 'valence_current', 'stressed_current'] # 'arousal_current', 'valence_current', 'stressed_current'
feature_dims = ['1D'] # 1D, 1D-stat, 2D
ds_info = get_dataset_dict('emowatch')

cluster_num = 0
for feature_dim in feature_dims:
    ds_info['feature-dim'] = feature_dim
    for label in labels:
        ds_info['chosen-label'] = label
        result = load_emowatch_finetuning(ds_info=ds_info, save_data=True, cluster_num=cluster_num, include_sleep=include_sleep)