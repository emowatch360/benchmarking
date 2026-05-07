"""
Inputs : numpy file path containing current features and labels, sleep data file path
Output : new numpy file path containing current features, sleep normalized features and labels
Sleep normalized features are computed by normalizing current features with sleep data (mean of all individual features) until the current day for each individual
i.e., do not include future days sleep data for normalization
"""

import os
import numpy as np
import pandas as pd
from data_config import get_dataset_dict
from ds_utils import validate_dimensions

cfg = get_dataset_dict('emowatch')
chosen_label = 'arousal_current'
feature_dim = '2D' # does not work with 1D-stat because the features don't exactly line up
data_path = f'/data/emowatch/emowatch/numpy_arrays_finetuning/2class/both_features28_epoch30/all_emowatch_{feature_dim}_{chosen_label}_finetuning.npz'
sleep_data_path = '/data/emowatch/emowatch/processed_sleep_data_all_filtered'
assert feature_dim in ['2D'], f'Invalid feature dimension: {feature_dim}'
assert 'sleep' not in data_path, 'Input data path should not contain sleep features'
output_path = data_path.replace('.npz', '_sleep_normalized.npz')
data = np.load(data_path)
pid_list = np.unique(data['str_pid_labels'])
day_list = [f'd{day:02d}' for day in range(1, 29)]
sleep_features_list = cfg['sleep-features-of-interest']
# remove features related to actigraphy
act_feature_list = ['Motion Intensity', 'Step Count', 'Total Count', 'Zero Crossing Count']
sleep_features_list = [feat for feat in sleep_features_list if feat not in act_feature_list]
# based on data['feature_names'], get the feature indices corresponding to sleep_features_list
np_feature_names = np.array(data['feature_names'])
feature_indices = [np.where(np_feature_names == feat)[0][0] for feat in sleep_features_list]
act_feature_indices = [np.where(np_feature_names == feat)[0][0] for feat in act_feature_list]
alpha = 0.4  # smoothing factor for moving average
eps=1e-6  # small value to avoid division by zero
# create a copy of the numpy array to store sleep normalized features
sleep_normalized_features, all_labels, all_values, all_prev_survey_values = [], [], [], []
all_dg_labels, all_str_pid_labels, all_pid_labels  = [], [], []
all_day_labels, all_session_labels, all_time_indices = [], [], []

for pid in pid_list:
    sleep_mean_until_day = None
    sleep_std_until_day = None
    for day in day_list:
        # filter features for current pid and day
        curr_indices = np.where((data['str_pid_labels'] == pid) & (data['day_labels'] == int(day[1:])))[0]
        # load sleep data for current pid
        sleep_file_path = os.path.join(sleep_data_path, f'{pid}-{day}-sleep-features.csv')
        if not os.path.exists(sleep_file_path):
            print(f'Sleep data file not found: {sleep_file_path}')
        else:
            # read the required columns (based on sleep features of interest) and compute mean across all non-nan values
            sleep_df = pd.read_csv(sleep_file_path)
            sleep_df = sleep_df[sleep_features_list]
            sleep_mean = sleep_df.mean(axis=0, skipna=True).values
            sleep_std = sleep_df.std(axis=0, skipna=True).values
            # compute a moving average of sleep means until the current day
            if sleep_mean_until_day is None:
                sleep_mean_until_day = sleep_mean
                sleep_std_until_day = sleep_std
            else:
                sleep_mean_until_day = alpha * sleep_mean + (1 - alpha) * sleep_mean_until_day
                sleep_std_until_day = alpha * sleep_std + (1 - alpha) * sleep_std_until_day
                
        if sleep_mean_until_day is None:
            print(f'No sleep data available until {day} for {pid}, skipping normalization for this day.')
            continue
        if np.any(np.isnan(sleep_mean_until_day)):
            print(f'Sleep mean contains NaN values for {pid} until {day}, skipping normalization for this day.')
            continue
        if len(curr_indices) == 0:
            print(f'No data available for {pid} on {day}, skipping.')
            continue
        assert len(feature_indices) == sleep_mean_until_day.shape[0], f'Expected {sleep_mean_until_day.shape[0]} features, got {data["features"].shape[2]}'
        assert np.any(np.isnan(sleep_std_until_day)) == False, f'Sleep std contains NaN values for {pid} until {day}'
        assert np.any(np.isnan(sleep_mean_until_day)) == False, f'Sleep mean contains NaN values for {pid} until {day}'
        # normalize current features with sleep means
        updated_features = data['features'][curr_indices]
        # update only non actigraphy features
        if feature_dim == '2D':
            updated_features[:, :, feature_indices] = (updated_features[:, :, feature_indices] - sleep_mean_until_day)/(sleep_std_until_day+eps)
            # updated_features[:, :, act_feature_indices] = np.log1p(updated_features[:, :, act_feature_indices])
        else:
            updated_features[:, feature_indices] = (updated_features[:, feature_indices] - sleep_mean_until_day)/(sleep_std_until_day+eps)
        sleep_normalized_features.append(updated_features)
        all_labels.append(data['labels'][curr_indices])
        all_values.append(data['values'][curr_indices])
        all_prev_survey_values.append(data['prev_survey_values'][curr_indices])
        all_dg_labels.append(data['dg_labels'][curr_indices])
        all_str_pid_labels.append(data['str_pid_labels'][curr_indices])
        all_pid_labels.append(data['pid_labels'][curr_indices])
        all_day_labels.append(data['day_labels'][curr_indices])
        all_session_labels.append(data['session_labels'][curr_indices])
        all_time_indices.append(data['time_indices'][curr_indices])

# save the new numpy array
sleep_normalized_features = np.vstack(sleep_normalized_features)
all_labels =  np.concatenate(all_labels)
all_values =  np.concatenate(all_values)
all_prev_survey_values =  np.concatenate(all_prev_survey_values)
all_dg_labels =  np.concatenate(all_dg_labels)
all_str_pid_labels =  np.concatenate(all_str_pid_labels)
all_pid_labels =  np.concatenate(all_pid_labels)
all_day_labels =  np.concatenate(all_day_labels)
all_session_labels =  np.concatenate(all_session_labels)
all_time_indices =  np.concatenate(all_time_indices)
result = {
    'features': sleep_normalized_features,
    'labels': all_labels,
    'values': all_values,
    'prev_survey_values': all_prev_survey_values,
    'dg_labels': all_dg_labels,
    'str_pid_labels': all_str_pid_labels,
    'pid_labels': all_pid_labels,
    'day_labels': all_day_labels,
    'session_labels': all_session_labels,
    'time_indices': all_time_indices,
    'feature_names': data['feature_names']
}
validate_dimensions(result, include_sleep=False)
np.savez(output_path, **result)


