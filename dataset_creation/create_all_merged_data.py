from merge_datasets import merge_and_save_datasets
from ..model_training.configs.shared_config import data_folder # type: ignore

data_folder = data_folder + 'numpy_arrays_finetuning/'
feature_type_list = ['2D','1D-stat'] #'2D' #'1D-stat'
num_class = 2
context_min_list = [30, 15]
feature_dim = 28
label_str_list = ['stressed_current', 'valence_current', 'arousal_current']
sample_threshold = 10 # minimum number of samples per participant to be considered 'good'
label_frac = 0.2 # minimum fraction of each class for a participant to be considered 'good'
filter_good_participants = False # whether to filter out participants with too few samples / poor label distribution
feature_normalization = False # whether features are normalized per participant
include_sleep_list = [False, True]

for feature_type in feature_type_list:
    for context_min in context_min_list:
        for label_str in label_str_list:
            for include_sleep in include_sleep_list:
                merge_and_save_datasets(
                    data_folder=data_folder,
                    feature_type=feature_type,
                    num_class=num_class,
                    context_min=context_min,
                    feature_dim=feature_dim,
                    label_str=label_str,
                    sample_threshold=sample_threshold,
                    label_frac=label_frac,
                    include_sleep=include_sleep,
                    filter_good_participants=filter_good_participants,
                    feature_normalization=feature_normalization,
                    save_path=None
                )