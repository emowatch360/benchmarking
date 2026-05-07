import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch.utils.data import random_split

def convert_to_minutes(timestamp, fmt='YYYY-MM-DDTHH:MM:SS.sss'):
    """
    Convert a timestamp string to the number of minutes since the start of the day.
    Supported formats:
      - 'YYYY-MM-DDTHH:MM:SS.sss'
      - 'YYYY/MM/DD HH:MM:SS'
      - 'HH:MM'
    Returns an integer count of minutes.
    """
    if fmt == 'YYYY-MM-DDTHH:MM:SS.sss':
        # Extract the time part 'HH:MM:SS'
        time_str = timestamp[11:19]
        hour, minute, _ = map(int, time_str.split(':'))
        total_minutes = hour * 60 + minute
    elif fmt == 'YYYY/MM/DD HH:MM:SS' or fmt == 'YYYY-MM-DD HH:MM:SS':
        time_str = timestamp.split(' ')[1]
        hour, minute, _ = map(int, time_str.split(':'))
        total_minutes = hour * 60 + minute
    elif fmt == 'YYYY/MM/DD HH:MM':
        time_str = timestamp.split(' ')[1]
        hour, minute = map(int, time_str.split(':'))
        total_minutes = hour * 60 + minute
    elif fmt == 'HH:MM':
        hour, minute = map(int, timestamp.split(':'))
        total_minutes = hour * 60 + minute
    else:
        raise ValueError(f"Incorrect timestamp format: {fmt}")
    return total_minutes

def create_windowed_data(features_mat, labels_arr=[], window_size=30, stride_length=10, start_idx=0):
    windowed_features = []
    windowed_time_indices = []
    feature_size = features_mat.shape[0]
    if labels_arr == []: # pretraining
        for idx in range(window_size, feature_size+1, stride_length):
            window = features_mat[idx-window_size:idx, :]
            if np.any(np.isnan(window)):
                continue
            # only choose windows with no NaN values
            windowed_features.append(window)
            windowed_time_indices.append(start_idx+idx-window_size) # saves the start time index of the window
            # windowed_masks.append(window_mask)
        windowed_features = np.array(windowed_features)
        windowed_time_indices = np.array(windowed_time_indices)
        # windowed_masks = np.array(windowed_masks)
    return windowed_features, windowed_time_indices#, windowed_masks

def shuffle_dataset(dataset, generator=None):
    assert generator != None, 'Generator is undefined!'
    total = len(dataset)
    indices = torch.randperm(total, generator=generator)
    shuffled_dataset = torch.utils.data.Subset(dataset, indices)
    return shuffled_dataset

def split_dataset(dataset, split_ratios={'train':0.6, 'val': 0.2}, generator=None):
    assert generator != None,  'Generator is undefined!'
    total = len(dataset)
    train_size = int(split_ratios['train'] * total)
    val_size = int(split_ratios['val'] * total)
    test_size = total - train_size - val_size
    return random_split(dataset, [train_size, val_size, test_size], generator=generator)

def map_value_to_label(value, bins):
	num_bins = len(bins)
	label = np.nan
	for idx in np.arange(num_bins-1):
		if idx == num_bins-2:
			if value >= bins[idx] and value <= bins[idx+1]:
				label = idx
				break
		if value >= bins[idx] and value < bins[idx+1]:
			label = idx
			break
	if label != label: # it's nan
		pass
		# print(f'Error: label = {label}, {value}')
	return label

def compute_discrete_labels(value_list, num_classes, min_val=0, max_val=100, label_type='continuous'):
	# print(value_list)
    if label_type == 'continuous':
        bins = np.arange(min_val, max_val, np.ceil((max_val-min_val)/num_classes), dtype=int)
        bins = np.append(bins, max_val)
        label_list = np.array([map_value_to_label(x, bins) for x in value_list])
    elif label_type == 'likert':
        if num_classes == 3:
            bins = [-np.inf, -0.5, 0.5, np.inf] # 0 is a separate class
        elif num_classes == 2:
            bins = [-np.inf, 0, np.inf] # 0 is considered positive
        else:
            raise ValueError(f'Unsupported number of classes for likert labels: {num_classes}')
        label_list = np.array([map_value_to_label(x, bins) for x in value_list])
    else:
        raise ValueError(f'Unknown label type: {label_type}')
    return label_list

def one_hot_encoder(in_array, num_unique_labels=-1):
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(in_array)
    # binary encode
    if num_unique_labels == -1:
        onehot_encoder = OneHotEncoder(sparse_output=False) #sparse=False -- modified !!
    else:
        assert np.size(np.unique(in_array)) <= num_unique_labels, 'Too many labels!'
        onehot_encoder = OneHotEncoder(sparse_output=False, categories=[np.arange(num_unique_labels)])
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    cls_onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return cls_onehot_encoded

def get_demographics_values(pid, demographic_df):
    pid_df = demographic_df[demographic_df['Subject'] == pid]
    dg_values = np.array([pid_df['Binary Gender'].astype(int), pid_df['Age'].astype(int)]).T
    return dg_values

def interpolate_data(raw_df, ds_info):
    if ds_info['interp']:
        interp_dict = ds_info['interp-prop']
        features_df = raw_df.interpolate(method=interp_dict['method'],
                                        axis=interp_dict['axis'],
                                        limit_direction=interp_dict['limit_direction'],
                                        limit=interp_dict['limit'])
    else:
        features_df = raw_df
    return features_df

def get_sleep_start_and_end_times(pid, day, dates_df, sleep_summary_df, day_list, ds_info):
    # based on chosen day, identify corresponding date to get sleep time indices
    # get current date
    curr_date = dates_df.loc[dates_df['pid'] == pid, day]
    assert len(curr_date) == 1, f'Expected 1 date for PID: {pid}, Day: {day}, but got {len(curr_date)}'
    curr_date = pd.to_datetime(curr_date.iloc[0]).normalize()
    total_minutes = ds_info['total-minutes']

    def get_relative_minutes(timestamp):
        event_time = pd.to_datetime(timestamp)
        relative_minutes = int((event_time - curr_date).total_seconds() // 60)
        return relative_minutes

    # get sleep time indices from sleep summary data
    sleep_row = sleep_summary_df[(sleep_summary_df['PID'] == pid)
                                    & (sleep_summary_df['Wake Day'] == day)]
    if sleep_row.empty:
        wake_time = ds_info['default-wake-time-index']
    else:
        assert len(sleep_row) == 1, f'Expected 1 sleep summary row for PID: {pid}, Day: {day}, but got {len(sleep_row)}'
        wake_time = get_relative_minutes(sleep_row['end_time'].iloc[0])
        wake_time = min(max(wake_time, 0), total_minutes)
    # get sleep start time
    # convert curr_date to datetime object
    next_date = curr_date + pd.Timedelta(days=1)
    next_day = f'd{int(day[1:])+1:02d}'
    # check if next_day is one of the valid days
    if next_day in day_list:
        next_day_from_df = dates_df.loc[dates_df['pid'] == pid, next_day].iloc[0]
        if next_date != pd.to_datetime(next_day_from_df): # next day not chronologically correct
            # next day date does not match
            sleep_time = ds_info['default-sleep-time-index']
            print(f'Warning: For PID: {pid}, Day: {day}, next day does not match! Using default sleep time index.')
            # raise ValueError('For C1 dataset, next day should always match!')
        else:
            sleep_row_next = sleep_summary_df[(sleep_summary_df['PID'] == pid)
                                            & (sleep_summary_df['Wake Day'] == next_day)]
            if sleep_row_next.empty:
                sleep_time = ds_info['default-sleep-time-index']
            else:
                assert len(sleep_row_next) == 1, f'Expected 1 sleep summary row for PID: {pid}, Day: {next_day}, but got {len(sleep_row_next)}'
                sleep_time = get_relative_minutes(sleep_row_next['start_time'].iloc[0])
    else:
        sleep_time = ds_info['default-sleep-time-index']
    return wake_time, sleep_time

# def get_sleep_clusters(sleep_cluster_path, pid, day,
#                        columns_of_interest):
#     sleep_df = pd.read_csv(sleep_cluster_path)
#     sleep_df_filtered = sleep_df[sleep_df['start_time'] != 'error']
#     subset_df = sleep_df_filtered[(sleep_df_filtered['PID'] == pid) & (sleep_df_filtered['Wake Day'] == day)].copy()
#     sleep_cluster_df = subset_df[columns_of_interest]
#     sleep_cluster_df = sleep_cluster_df.dropna()
#     if sleep_cluster_df.empty:
#         return np.array([np.nan])
#     assert len(sleep_cluster_df) == 1
#     sleep_cluster_features = sleep_cluster_df.to_numpy()[0]
#     return sleep_cluster_features

def get_sleep_statistics(sleep_stat_path, pid, wake_day,
                         night_minutes=960,
                         total_minutes=1440,
                         columns_of_interest=['duration', 'Rem_per', 'Deep_per', 'Light_per', 'Awake_per']
                         ):
    sleep_df = pd.read_csv(sleep_stat_path)
    sleep_df_filtered = sleep_df[sleep_df['start_time'] != 'error']
    subset_df = sleep_df_filtered[(sleep_df_filtered['PID'] == pid) & (sleep_df_filtered['Wake Day'] == wake_day)].copy()
    sleep_stat_df = subset_df[columns_of_interest]
    sleep_stat_df = sleep_stat_df.dropna()
    if sleep_stat_df.empty:
        return np.array([np.nan])
    assert len(sleep_stat_df) == 1
    if 'start_time' in columns_of_interest:
        # convert start time to minutes
        # sleep_stat_df['start_time'] = sleep_stat_df['start_time'].apply(lambda x: convert_to_minutes(x, fmt='YYYY/MM/DD HH:MM'))
        sleep_stat_df['start_time'] = sleep_stat_df['start_time'].apply(lambda x: convert_to_minutes(x, fmt='YYYY/MM/DD HH:MM:SS')) # for corrected_hrv_numpy_arrays
        if sleep_stat_df['start_time'].iloc[0] > night_minutes:
            sleep_stat_df['start_time'] = sleep_stat_df['start_time']-total_minutes
    if 'end_time' in columns_of_interest:
        # convert start time to minutes
        # sleep_stat_df['end_time'] = sleep_stat_df['end_time'].apply(lambda x: convert_to_minutes(x, fmt='YYYY/MM/DD HH:MM'))
        sleep_stat_df['end_time'] = sleep_stat_df['end_time'].apply(lambda x: convert_to_minutes(x, fmt='YYYY/MM/DD HH:MM:SS')) # for corrected_hrv_numpy_arrays
        assert sleep_stat_df['end_time'].iloc[0] < night_minutes, 'End time is not less than night minutes!'
    if 'duration' in columns_of_interest:
        sleep_stat_df = sleep_stat_df.astype(float)
        sleep_stat_df['duration'] = sleep_stat_df['duration']/3600. # convert seconds to hour
    sleep_stat_features = sleep_stat_df.to_numpy()[0]
    return sleep_stat_features

def get_sleep_features(sleep_feature_path,
                       pid, wake_day,
                       feature_dim,
                       features_of_interest=['RMSSD'],
                       count_feature_name_list=[],
                       num_chunks=30,
                       ds_info=None):
    file_path = os.path.join(sleep_feature_path, f'{pid}-{wake_day}-sleep-features.csv')
    if not os.path.exists(file_path):
        return np.array([np.nan])
    sleep_df = pd.read_csv(file_path)
    if len(sleep_df) == 0:
         return np.array([np.nan])
    raw_sleep_features_df = sleep_df[features_of_interest]
    # take count-based features and apply log1p transformation
    for count_feat in count_feature_name_list:
        if count_feat in raw_sleep_features_df.columns:
            raw_sleep_features_df.loc[:, count_feat] = np.log1p(raw_sleep_features_df[count_feat])
    # perform interpolation
    sleep_features_df = interpolate_data(raw_df=raw_sleep_features_df, ds_info=ds_info)
    # 1D features are computed over the entire sleep period
    sleep_features_1d = compute_descriptive_stats(sleep_features_df)
    if np.any(np.isnan(sleep_features_1d)):
        return np.array([np.nan])
    assert sleep_features_1d.shape[0] == len(features_of_interest)*5
    if '1D' in feature_dim: # unmatched !!
        return sleep_features_1d
    # 2D features are computed over multiple chunks of sleep
    # sleep_features_2d = []
    num_indices = len(sleep_features_df)
    chunk_indices = np.linspace(0, num_indices, num_chunks+1, endpoint=True, dtype=int)
    sleep_features_list = []
    for idx in np.arange(num_chunks):
        start_idx = chunk_indices[idx]
        end_idx = chunk_indices[idx+1]
        subset_df = sleep_features_df.iloc[start_idx:end_idx]
        sleep_features_idx = compute_descriptive_stats(subset_df, statistics='mean')
        if np.any(np.isnan(sleep_features_idx)):
            return np.array([np.nan])
        sleep_features_list.append(sleep_features_idx)
    sleep_features_2d = np.vstack(sleep_features_list)
    assert sleep_features_2d.shape[0] == num_chunks
    assert sleep_features_2d.shape[1] == len(features_of_interest)
    if '2D' in feature_dim:
        return sleep_features_2d

def validate_dimensions(result, include_sleep):
    n_samples = result['features'].shape[0]
    assert result['labels'].shape[0] == n_samples, 'labels do not match!'
    assert result['values'].shape[0] == n_samples, 'values do not match!'
    assert result['session_labels'].shape[0] == n_samples, 'session labels do not match!'
    assert result['time_indices'].shape[0] == n_samples, 'time indices do not match!'
    assert result['prev_survey_values'].shape[0] == n_samples, 'prev survey values do not match!'
    assert result['str_pid_labels'].shape[0] == n_samples, 'pid labels do not match!'
    assert result['pid_labels'].shape[0] == n_samples, 'pid labels do not match!'
    assert result['day_labels'].shape[0] == n_samples, 'day labels do not match!'
    assert result['dg_labels'].shape[0] == n_samples, 'dg labels do not match!'
    if include_sleep:
        assert result['sleep_statistics'].shape[0] == n_samples, 'sleep statistics do not match!'
        assert result['sleep_features'].shape[0] == n_samples, 'sleep features do not match!'
    return True

def validate_dimensions_pretraining(result):
    n_samples = result['features'].shape[0]
    assert result['time_indices'].shape[0] == n_samples, 'time indices do not match!'
    assert result['str_pid_labels'].shape[0] == n_samples, 'str pid labels do not match!'
    assert result['pid_labels'].shape[0] == n_samples, 'pid labels do not match!'
    assert result['day_labels'].shape[0] == n_samples, 'day labels do not match!'
    return True

def validate_dimensions_pretraining_sleep(result):
    n_samples = result['sleep_features'].shape[0]
    assert result['str_pid_labels'].shape[0] == n_samples, 'str pid labels do not match!'
    assert result['pid_labels'].shape[0] == n_samples, 'pid labels do not match!'
    assert result['day_labels'].shape[0] == n_samples, 'day labels do not match!'
    return True

def compute_descriptive_stats(data_matrix, statistics='all'):
    nan_map = np.all(np.isnan(data_matrix), axis=0)
    if np.any(nan_map):
        return np.array([np.nan])
    if statistics == 'all':
        stat_list = np.hstack((np.nanmean(data_matrix, axis=0),
                                np.nanmedian(data_matrix, axis=0),
                                np.nanmax(data_matrix, axis=0),
                                np.nanmin(data_matrix, axis=0),
                                np.nanstd(data_matrix, axis=0)))
    elif statistics == 'mean':
        stat_list = np.nanmean(data_matrix, axis=0)
    else:
        print('Error: Unknown statistics')
        sys.exit(1)
    return stat_list

def create_features_with_stride(feature_mat, stride_dict):
    N = feature_mat.shape[0]
    window_len = stride_dict['window-length']
    hop_len = stride_dict['hop-length']
    selected_feature_list = []
    for idx in np.arange(0, N-window_len+1, hop_len):
        feature_mat_idx = feature_mat[idx:idx+window_len]
        if np.any(np.isnan(feature_mat_idx)):
            continue
        selected_feature_list.append(feature_mat_idx)
    feature_mat_3d = np.array(selected_feature_list)
    return feature_mat_3d # becomes a 3D feature 

def get_detailed_feature_names(feature_name_list, feature_dim, feature_type='current'):
    prefix = ''
    if feature_type == 'sleep':
        prefix = 'sl-'
    assert feature_dim in ['1D-stat', '2D', '2D-stride'], 'Unknown feature dimension!'
    if feature_dim != '1D-stat':
        detailed_features = [prefix+feature for feature in feature_name_list]
        return detailed_features
    # feature_type: current or sleep
    N = len(feature_name_list)
    detailed_features = ['' for i in range(N*5)]
    for idx, feature in enumerate(feature_name_list):
        detailed_features[idx] = prefix+feature+' mu' # mean
        detailed_features[idx+N] = prefix+feature+' md' # median
        detailed_features[idx+2*N] = prefix+feature+' mx' # max
        detailed_features[idx+3*N] = prefix+feature+' mn' # min
        detailed_features[idx+4*N] = prefix+feature+' std' # std dev
    return detailed_features

# def compute_descriptive_stats(data_matrix):
# 	stat_list = np.hstack((data_matrix.mean(axis=0),
# 							data_matrix.median(axis=0),
# 							# data_matrix.mode(axis=0).iloc[0],
# 							data_matrix.max(axis=0),
# 							data_matrix.min(axis=0),
# 							data_matrix.std(axis=0)))
# 	return stat_list

# def get_sleep_features(pid, day,
#                        feature_dim,
#                        sleep_feature_path,
#                        features_of_interest=['RMSSD']
#                        num_chunks=30):
#     num_features = 2 + len(features_of_interest)*5
#     nan_array = np.nan * np.ones(num_features)
#     prev_day = 'd'+f'{int(day[1:])-1:02d}'
#     file_path = os.path.join(sleep_feature_path, pid, prev_day, 'hrv_sleep.csv')
#     if not os.path.exists(file_path):
#         return nan_array
#     sleep_df = pd.read_csv(file_path)
#     # only consider sleep features that are in the HRV_sleep file
#     sleep_features = [feature for feature in features_of_interest if feature in sleep_df.columns]
#     if len(sleep_df) == 0:
#         return nan_array
#     sleep_df.replace(-200, np.nan, inplace=True) # convert all missing values to NaN
#     # Invalid time ranges
#     start_from = time(4, 0)
#     start_to = time(18, 0)
#     end_from = time(12, 0)
#     end_to = time(23, 59, 59)
#     sleep_start_time = datetime.strptime(sleep_df['Start Time'][0], time_fmt)
#     sleep_end_time = datetime.strptime(sleep_df['Start Time'][len(sleep_df)-1], time_fmt)
#     if (start_from <= sleep_start_time.time() <= start_to) or (end_from <= sleep_end_time.time() <= end_to):
#         return nan_array
#     else:
#         sleep_duration = (sleep_end_time-sleep_start_time).total_seconds()/60.0 # in mins
#         feature_stats = compute_descriptive_stats(sleep_df[sleep_features])
#         if np.any(np.isnan(feature_stats)):
#             return nan_array
#     # find the time corresponding to when feature-of-interest is maximum
#     max_feature_time_str = sleep_df['Start Time'][np.argmax(sleep_df[frac_time_feature])]
#     max_feature_time = datetime.strptime(max_feature_time_str, time_fmt)
#     frac_time = ((max_feature_time-sleep_start_time).total_seconds()/60.0) / sleep_duration
#     sleep_feature_list = np.concatenate((np.array([sleep_duration, frac_time]),
#                                    feature_stats))
#     return sleep_feature_list
