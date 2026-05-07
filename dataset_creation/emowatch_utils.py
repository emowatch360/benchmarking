import os
import sys
import numpy as np
import pandas as pd
from ds_utils import *
from data_config import *
from datetime import timedelta

def load_emowatch_labels(ds_info, chosen_pid, chosen_day):
    chosen_label = ds_info['chosen-label']
    # epoch_time = ds_info['duration']
    num_class = ds_info['num-classes']
    label_path = ds_info['label-path']
    label_df = pd.read_csv(label_path)
    subset_df = label_df[(label_df['PID']==chosen_pid) & (label_df['Day'].str.startswith(chosen_day))]
    subset_values = subset_df[chosen_label]
    subset_labels = compute_discrete_labels(subset_values, num_class, label_type=ds_info['label-type'])
    subset_time_index = np.array(subset_df['Time Index'])
    subset_session = np.array(subset_df['Session'])
    select_map =  ~np.isnan(subset_labels) # can handle with time_index is less than epoch_time
    # select_map = np.logical_and(subset_time_index >= epoch_time, ~np.isnan(subset_labels))
    # print(f'Removed Timestamps: {subset_time_index.size-sum(select_map)}')
    subset_time_index = subset_time_index[select_map]
    subset_labels = subset_labels[select_map]
    subset_values = subset_values[select_map]
    subset_session = subset_session[select_map]
    assert subset_labels.size == subset_values.size
    assert subset_time_index.size == subset_values.size
    assert subset_session.size == subset_values.size
    return subset_values, subset_labels, subset_time_index, subset_session

def load_emowatch_finetuning(ds_info, save_data=False, cluster_num=-1, include_sleep=False, dataset_country='both'):
    # read the labels, choose features at corresponding timestamps
    data_folder = ds_info['data-folder']
    dg_df = pd.read_csv(ds_info['demographics-path'])
    dataset_country = ds_info['dataset-country']
    pid_list = ds_info['pid-list']
    day_list = ds_info['day-list']
    feature_name_list = ds_info['features-of-interest']
    count_feature_name_list = ds_info['count-based-features']
    epoch_time = ds_info['duration']
    toff = ds_info['time-offset']
    feature_dim = ds_info['feature-dim']
    chosen_label = ds_info['chosen-label']
    total_minutes = ds_info['total-minutes']
    night_minutes = ds_info['night-minutes']
    print('='*50)
    print(f'Feature Dim: {feature_dim}, Chosen Label: {chosen_label}')
    feature_folder = ds_info['processed-feature-folder']
    # determining cluster information
    # cluster_df = pd.read_csv(ds_info['cluster-path'])
    # initializations
    all_features, all_labels, all_values, all_prev_survey_values = [], [], [], []
    all_dg_labels, all_str_pid_labels, all_pid_labels, all_day_labels, all_session_labels  = [], [], [], [], []
    all_time_indices, all_sleep_features, all_sleep_statistics = [], [], []
    success = 0
    for pid_idx, chosen_pid in enumerate(pid_list):
        print(f'PID: {chosen_pid}')
        prev_survey_value = 50 # default
        dg_values = get_demographics_values(pid=chosen_pid, demographic_df=dg_df)
        # baseline_feature_vec = compute_baseline_features(ds_info, chosen_pid)
        for chosen_day in day_list:
            if cluster_num != -1:
                sub_cluster_df = cluster_df[(cluster_df['PID'] == chosen_pid) 
                                            & (cluster_df['Wake Day'] == chosen_day)]
                assert len(sub_cluster_df) <= 1, f'Expected max 1 row for PID: {chosen_pid}, Day: {chosen_day}, but got {len(sub_cluster_df)}'
                if sub_cluster_df.empty or (sub_cluster_df['Cluster'] != cluster_num).any():
                    continue
            value_list, labels_list, time_stamps_list, session_list = load_emowatch_labels(ds_info,
                                                                 chosen_pid, chosen_day)
            # print(time_stamps_list)
            data_path = os.path.join(feature_folder, f'{chosen_pid}-{chosen_day}-features.csv')
            raw_features_df = pd.read_csv(data_path)
            raw_features_df = raw_features_df[feature_name_list]
            # take the count-based features and apply log1p transformation
            for count_feat in count_feature_name_list:
                if count_feat in raw_features_df.columns:
                    raw_features_df.loc[:, count_feat] = np.log1p(raw_features_df[count_feat])
            for value, label, time_idx, session in zip(value_list, labels_list,
                                                       time_stamps_list, session_list):
                if np.isnan(label): # it's nan
                    print('Error: Invalid label')
                    continue
                    # sys.exit(1)
                # incorporate time offset
                assert (time_idx+1-toff) <= total_minutes, 'Last index exceeds max length!'
                if time_idx-epoch_time+1-toff >= 0:
                    time_df = raw_features_df.loc[(time_idx-epoch_time+1-toff):(time_idx-toff)] # loc is inclusive of the last index
                else:
                    assert session in ['evening', 'bedtime'], f'Time index should be greater than epoch time for {session}!'
                    # read previous day's data and today's data
                    prev_day = f'd{int(chosen_day[1:])-1:02d}'
                    if prev_day not in ds_info['day-list']:
                        print(f'Previous day {prev_day} not in day list!')
                        continue
                    # check if prev day and next day are consecutive days
                    if dataset_country == 'c2':
                        prev_date = c2_dates_df.loc[c2_dates_df['pid'] == chosen_pid, prev_day].iloc[0]
                        curr_date = c2_dates_df.loc[c2_dates_df['pid'] == chosen_pid, chosen_day].iloc[0]
                        assert pd.to_datetime(prev_date) == (pd.to_datetime(curr_date) - timedelta(days=1)), 'dates are not consecutive!'
                    prev_data_path = os.path.join(feature_folder, f'{chosen_pid}-{prev_day}-features.csv')
                    prev_features_df = pd.read_csv(prev_data_path)
                    prev_features_df = prev_features_df[feature_name_list]
                    # concatenate previous day's data with current day's data
                    # modify the indices of the previous day -- 1439 should be -1, 1439 should be -2, and so on
                    prev_features_df.index = prev_features_df.index - total_minutes
                    combined_df = pd.concat([prev_features_df, raw_features_df], axis=0)
                    time_df = combined_df.loc[(time_idx-epoch_time+1-toff):(time_idx-toff)] # loc is inclusive of the last index
                features_df = interpolate_data(raw_df=time_df, ds_info=ds_info)
                feature_mat = features_df.to_numpy()
                assert (feature_mat.shape[0] == epoch_time), 'Feature matrix should have the same number of rows as epoch time!'
                # feature_mat = feature_mat - baseline_feature_vec
                if feature_dim not in ['1D', '2D', '1D-stat', '2D-stride']:
                    print(f'Incorrect feature dim: {feature_dim}!')
                    sys.exit(1)
                if feature_dim == '2D':
                    if (np.any(np.isnan(feature_mat))):
                        prev_survey_value = value
                        continue
                    num_entries = 1
                elif feature_dim == '1D':
                    time_mask = ~np.any(np.isnan(feature_mat), axis=1)
                    assert time_mask.size == feature_mat.shape[0]
                    feature_mat = feature_mat[time_mask, :]
                    if feature_mat.shape[0] == 0:
                        prev_survey_value = value
                        continue
                    num_entries = feature_mat.shape[0]
                elif feature_dim == '2D-stride':
                    feature_mat = create_features_with_stride(feature_mat,
                                                              ds_info['2D-stride'])
                    if feature_mat.shape[0] == 0:
                        prev_survey_value = value
                        continue
                    num_entries = feature_mat.shape[0]
                else: # 1D-stat
                    # input is 2D, output is 1D
                    # if atleast one feature is nan at all time points
                    if np.any(np.all(np.isnan(feature_mat), axis=0)): # unmatched criteria!!
                    # if (np.any(np.isnan(feature_mat))): # use same criteria as 2D matrix
                        # print(np.where(np.all(np.isnan(feature_mat), axis=0)))
                        # print(f'{chosen_day}: 1D current: Feature has all NaN values for at least one feature!')
                        prev_survey_value = value
                        continue
                    feature_mat = compute_descriptive_stats(feature_mat)
                    num_entries = 1
                # read sleep features
                if include_sleep:
                    if (session not in ['bedtime', 'evening']) or (time_idx > night_minutes):
                        wake_day = chosen_day
                    else:
                        # choose the previous day to be wake_day
                        wake_day = f'd{int(chosen_day[1:])-1:02d}' # accounts for surveys that were filled out past midnight
                        if wake_day not in ds_info['day-list']:
                            print(f'Wake day {wake_day} not in day list; Time Index: {time_idx}!')
                            continue
                        if dataset_country == 'c2':
                            # check if prev day and next day are consecutive days
                            prev_date = c2_dates_df.loc[c2_dates_df['pid'] == chosen_pid, wake_day].iloc[0]
                            curr_date = c2_dates_df.loc[c2_dates_df['pid'] == chosen_pid, chosen_day].iloc[0]
                            assert pd.to_datetime(prev_date) == (pd.to_datetime(curr_date) - timedelta(days=1)), 'dates are not consecutive!'
                    sleep_statistics = get_sleep_statistics(sleep_stat_path=ds_info['sleep-stat-path'],
                                                        pid=chosen_pid,
                                                        wake_day=wake_day,
                                                        night_minutes=night_minutes,
                                                        total_minutes=total_minutes,
                                                        columns_of_interest=ds_info['sleep-stat-features'])
                    if np.any(np.isnan(sleep_statistics)):
                        # print(f'{chosen_day}: Sleep statistics has NaN values!')
                        continue
                    sleep_features = get_sleep_features(sleep_feature_path=ds_info['sleep-feat-path'],
                                                        pid=chosen_pid,
                                                        wake_day=wake_day,
                                                        feature_dim=feature_dim,
                                                        features_of_interest=ds_info['sleep-features-of-interest'],
                                                        count_feature_name_list=count_feature_name_list,
                                                        num_chunks=ds_info['sleep-2d-chunks'],
                                                        ds_info=ds_info) # ds_info sent for interpolation
                    if np.any(np.isnan(sleep_features)):
                        # print(f'{chosen_day}: Sleep features has NaN values!')
                        continue
                else:
                    if (session not in ['bedtime', 'evening']) or (time_idx > night_minutes):
                        wake_day = chosen_day
                    else:
                        # choose the previous day to be wake_day
                        wake_day = f'd{int(chosen_day[1:])-1:02d}' # accounts for surveys that were filled out past midnight
                all_features.append(feature_mat)
                all_labels.extend(np.repeat(label, num_entries))
                all_values.extend(np.repeat(value, num_entries))
                all_prev_survey_values.extend(np.repeat(prev_survey_value, num_entries))
                all_session_labels.extend(np.repeat(session, num_entries))
                all_day_labels.extend(np.repeat(int(wake_day[1:]), num_entries)) # change this to wake_day
                # all_pid_labels.extend(np.repeat(int(chosen_pid[3:]), num_entries))
                all_str_pid_labels.extend(np.repeat(chosen_pid, num_entries))
                all_pid_labels.extend(np.repeat(pid_idx+1, num_entries))
                all_dg_labels.extend(np.tile(dg_values, (num_entries, 1)))
                all_time_indices.extend(np.repeat(time_idx, num_entries))
                if include_sleep:
                    all_sleep_statistics.extend(np.tile(sleep_statistics, (num_entries, 1)))
                    if '1D' in feature_dim:
                        all_sleep_features.extend(np.tile(sleep_features, (num_entries, 1)))
                    elif '2D' in feature_dim:
                        all_sleep_features.extend(np.tile(sleep_features, (num_entries, 1, 1)))
                # assumes the day list goes in order (i.e., d01, d02, ...)
                prev_survey_value = value
    print(len(all_features))
    if len(all_features) == 0:
        return np.array([]), np.array([])
    all_prev_survey_values = np.array(all_prev_survey_values).reshape(-1,1)
    all_time_indices = np.array(all_time_indices).reshape(-1,1)
    if feature_dim == '1D':
        all_features = np.concatenate(all_features)
        # all_features = np.hstack((all_time_indices, all_features, all_prev_survey_values))
    elif feature_dim == '2D':
        all_features = np.array(all_features) # 2D does not include prev survey values
    elif feature_dim == '2D-stride':
        all_features = np.concatenate(all_features, axis=0) # 2D-stride does not include prev survey values
    else: # 1D-stat
        all_features = np.array(all_features)
        # all_features = np.hstack((all_time_indices, all_features, all_prev_survey_values))
    result = {
        'features': all_features,
        'labels': np.array(all_labels), # binary
        'values': np.array(all_values),
        'session_labels': np.array(all_session_labels),
        'time_indices': np.array(all_time_indices),
        'prev_survey_values': np.array(all_prev_survey_values), # prev survey label computation is not fully accurate when sleep data is invalid
        'day_labels': np.array(all_day_labels),
        'pid_labels': np.array(all_pid_labels),
        'str_pid_labels': np.array(all_str_pid_labels),
        'dg_labels': np.array(all_dg_labels),
        'feature_names': get_detailed_feature_names(feature_name_list, feature_dim, feature_type='current')
    }
    if include_sleep:
        result.update({
            'sleep_statistics': np.array(all_sleep_statistics),
            'sleep_features': np.array(all_sleep_features),
            'sleep_statistic_names': ds_info['sleep-stat-features'],
            'sleep_feature_names': get_detailed_feature_names(ds_info['sleep-features-of-interest'], feature_dim, feature_type='sleep')
        })
    assert validate_dimensions(result, include_sleep)
    if save_data:
        save_path = os.path.join(ds_info['save-folder'], f'all_emowatch_{feature_dim}_{chosen_label}_finetuning.npz')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if cluster_num != -1:
            save_path = save_path[:-14]+f'{ds_info['cluster-feature']}_cluster{cluster_num}_'+save_path[-14:]
        if include_sleep:
            save_path = save_path[:-14]+'all_sleep_features_'+save_path[-14:]
        np.savez(save_path, **result)
    return result['features'], result['labels']

# used for loading current features for pretraining
# identify non-sleep windows by taking into account sleep time indices
def load_emowatch_pretraining(ds_info, save_data=False):
    pid_list = ds_info['pid-list']
    day_list = ds_info['day-list']
    feature_name_list = ds_info['features-of-interest']
    count_feature_name_list = ds_info['count-based-features']
    feature_folder = ds_info['processed-feature-folder']
    dataset_country = ds_info['dataset-country']
    total_minutes = ds_info['total-minutes']
    dates_df = pd.read_csv(ds_info[f'{dataset_country}-dates-path'])
    sleep_summary_df = pd.read_csv(ds_info['sleep-stat-path'])
    epoch_time = ds_info['duration']
    all_windowed_features = []
    all_str_pid_labels, all_pid_labels, all_day_labels, all_time_indices  = [], [], [], []
    print('='*50)
    print(f'Dataset Country: {dataset_country}, Epoch Duration: {epoch_time} minutes')
    for pidx, chosen_pid in enumerate(pid_list):
        print(f'PID: {chosen_pid}')
        for chosen_day in day_list:
            data_path = os.path.join(feature_folder, f'{chosen_pid}-{chosen_day}-features.csv')
            raw_features_df = pd.read_csv(data_path)
            raw_features_df = raw_features_df[feature_name_list]
            # take the count-based features and apply log1p transformation
            for count_feat in count_feature_name_list:
                if count_feat in raw_features_df.columns:
                    raw_features_df.loc[:, count_feat] = np.log1p(raw_features_df[count_feat])
            features_df = interpolate_data(raw_df=raw_features_df, ds_info=ds_info)
            wake_time_idx, sleep_time_idx = get_sleep_start_and_end_times(pid=chosen_pid, day=chosen_day,
                                                                              dates_df=dates_df, sleep_summary_df=sleep_summary_df,
                                                                              day_list=day_list, ds_info=ds_info)
            # extract only the non-sleep portion
            feature_mat = features_df.to_numpy()
            if sleep_time_idx <= wake_time_idx:
                print(f'Warning: For PID: {chosen_pid}, Day: {chosen_day}, wake time {wake_time_idx} is not earlier than sleep time {sleep_time_idx}. Skipping day.')
                continue
            # including next day data depending on sleep time
            # these are also currently labelled under previous day for convenience sake
            if sleep_time_idx > total_minutes:
                next_day = f'd{int(chosen_day[1:])+1:02d}'
                if next_day in day_list:
                    next_data_path = os.path.join(feature_folder, f'{chosen_pid}-{next_day}-features.csv')
                    if os.path.exists(next_data_path):
                        next_features_df = pd.read_csv(next_data_path)
                        next_features_df = next_features_df[feature_name_list]
                        for count_feat in count_feature_name_list:
                            if count_feat in next_features_df.columns:
                                next_features_df.loc[:, count_feat] = np.log1p(next_features_df[count_feat])
                        next_features_df = interpolate_data(raw_df=next_features_df, ds_info=ds_info)
                        next_feature_mat = next_features_df.to_numpy()
                        feature_mat = np.concatenate(
                            [feature_mat[wake_time_idx:, :], next_feature_mat[:sleep_time_idx-total_minutes, :]],
                            axis=0
                        )
                    else:
                        feature_mat = feature_mat[wake_time_idx:, :]
                else:
                    feature_mat = feature_mat[wake_time_idx:, :]
            else:
                feature_mat = feature_mat[wake_time_idx:sleep_time_idx, :]
            # doing 10 minute stride for 30 minute epochs
            windowed_features, windowed_time_indices = create_windowed_data(feature_mat,
                                                                            window_size=epoch_time,
                                                                            stride_length=epoch_time//3,
                                                                            start_idx=wake_time_idx)
            num_windows = windowed_features.shape[0]
            # populate other features
            all_str_pid_labels.extend(np.repeat(chosen_pid, num_windows))
            all_pid_labels.extend(np.repeat(pidx+1, num_windows))
            all_day_labels.extend(np.repeat(int(chosen_day[1:]), num_windows))
            all_time_indices.extend(windowed_time_indices)
            if np.size(windowed_features) == 0:
                continue
            all_windowed_features.append(windowed_features)
            # print(windowed_features.shape)
    if len(all_windowed_features) == 0:
        print('No pretraining data found!')
        return
    data_mat = np.concatenate(all_windowed_features, axis=0)
    print(f'Pretraining data shape: {data_mat.shape}')
    if save_data:
        result = {
            'features': data_mat,
            'time_indices': np.array(all_time_indices),
            'day_labels': np.array(all_day_labels),
            'pid_labels': np.array(all_pid_labels),
            'str_pid_labels': np.array(all_str_pid_labels),
            'feature_names': get_detailed_feature_names(feature_name_list, ds_info['feature-dim'], feature_type='current')
        }
        validate_dimensions_pretraining(result)
        # save time indices, pid_labels, str_pid_labels, day_labels
        # run assertions to check dimensions
        save_path = os.path.join(ds_info['save-folder'], f'{dataset_country}_emowatch_pretraining.npz')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(save_path, **result)

def load_emowatch_pretraining_sleep(ds_info, save_data=False):
    pid_list = ds_info['pid-list']
    day_list = ds_info['day-list']
    count_feature_name_list = ds_info['count-based-features']
    dataset_country = ds_info['dataset-country']
    epoch_time = ds_info['duration']
    feature_dim = ds_info['feature-dim']
    assert feature_dim == '2D', 'Pretraining with sleep features is currently only implemented for 2D feature dimension!'
    all_sleep_features = []
    all_str_pid_labels, all_pid_labels, all_day_labels  = [], [], []
    print('='*50)
    print(f'Dataset Country: {dataset_country}, Epoch Duration: {epoch_time} minutes')
    for pidx, chosen_pid in enumerate(pid_list):
        print(f'PID: {chosen_pid}')
        for chosen_day in day_list:
            sleep_features = get_sleep_features(sleep_feature_path=ds_info['sleep-feat-path'],
                                                pid=chosen_pid,
                                                wake_day=chosen_day,
                                                feature_dim=feature_dim,
                                                features_of_interest=ds_info['sleep-features-of-interest'],
                                                count_feature_name_list=count_feature_name_list,
                                                num_chunks=ds_info['sleep-2d-chunks'],
                                                ds_info=ds_info) # ds_info sent for interpolation
            if np.any(np.isnan(sleep_features)):
                # print(f'{chosen_day}: Sleep features has NaN values!')
                continue
            all_sleep_features.append(sleep_features)
            # populate other features
            all_str_pid_labels.append(chosen_pid)
            all_pid_labels.append(pidx+1)
            all_day_labels.append(int(chosen_day[1:]))
    if len(all_sleep_features) == 0:
        print('No pretraining sleep data found!')
        return
    data_mat = np.array(all_sleep_features)
    print(f'Pretraining sleep feature data shape: {data_mat.shape}')
    if save_data:
        result = {
            'sleep_features': data_mat,
            'day_labels': np.array(all_day_labels),
            'pid_labels': np.array(all_pid_labels),
            'str_pid_labels': np.array(all_str_pid_labels),
            'sleep_feature_names': get_detailed_feature_names(ds_info['sleep-features-of-interest'], feature_dim, feature_type='sleep')
        }
        validate_dimensions_pretraining_sleep(result)
        # save time indices, pid_labels, str_pid_labels, day_labels
        # run assertions to check dimensions
        save_path = os.path.join(ds_info['save-folder'], f'{dataset_country}_emowatch_pretraining_sleep.npz')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(save_path, **result)   
