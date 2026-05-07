import os
import numpy as np
import pandas as pd

from data_config import get_dataset_dict
from ds_utils import convert_to_minutes, get_sleep_start_and_end_times
from ..model_training.configs.shared_config import data_folder # type: ignore

FIXED_DAYS = ['d01', 'd02']
MIN_VALID_MINUTES = 60
SAVE_PATH = data_folder + 'numpy_arrays_pretraining/2class/both_features28_epoch30/zscore_params.npz'
DATASET_COUNTRIES = ['c1', 'c2']
SLEEP_STAT_FEATURE_NAMES = [
    'duration', 'Rem_per', 'Deep_per', 'Light_per', 'Awake_per',
    'start_time', 'end_time'
]
DG_FEATURE_NAMES = ['gender', 'age']
TIME_FEATURE_NAMES = ['time_index']
GENDER_BERNOULLI_P = 0.5
AGE_UNIFORM_MIN = 21
AGE_UNIFORM_MAX = 40
TIME_UNIFORM_MIN = 0
TIME_UNIFORM_MAX = 1439

PID_TO_REMOVE = [
    'MM_12', 'c2com02', 'c2com16', 'pid02', 'pid04', 'pid05', 'pid07', 'pid11',
    'pid14', 'pid16', 'pid18', 'pid24', 'pid28', 'pid29', 'pid30', 'pid33',
    'pid34', 'pid32'
]


def build_ds_info(dataset_country):
    ds_info = get_dataset_dict('emowatch')
    ds_info['dataset-country'] = dataset_country
    if dataset_country == 'c1':
        orig_pid_list = ['MM_' + f'{pid:02d}' for pid in range(1, 46)]
        ds_info['label-type'] = 'continuous'
    elif dataset_country == 'c2':
        orig_pid_list = ['c2com' + f'{idx:02d}' for idx in range(1, 20)]
        orig_pid_list += ['pid' + f'{idx:02d}' for idx in range(1, 35)]
        ds_info['label-type'] = 'likert'
    else:
        raise ValueError(f'Unsupported dataset country: {dataset_country}')
    ds_info['pid-list'] = [pid for pid in orig_pid_list if pid not in PID_TO_REMOVE]
    return ds_info


def load_feature_dataframe(file_path, feature_name_list, count_feature_name_list, ds_info):
    if not os.path.exists(file_path):
        return None
    raw_df = pd.read_csv(file_path)
    if len(raw_df) == 0:
        return None
    feature_df = raw_df[feature_name_list].copy()
    for count_feat in count_feature_name_list:
        if count_feat in feature_df.columns:
            feature_df.loc[:, count_feat] = np.log1p(feature_df[count_feat])
    return feature_df


def get_non_sleep_dataframe(pid, day, ds_info, dates_df, sleep_summary_df):
    feature_folder = ds_info['processed-feature-folder']
    feature_name_list = ds_info['features-of-interest']
    count_feature_name_list = ds_info['count-based-features']
    total_minutes = ds_info['total-minutes']
    day_list = ds_info['day-list']

    curr_path = os.path.join(feature_folder, f'{pid}-{day}-features.csv')
    curr_df = load_feature_dataframe(
        file_path=curr_path,
        feature_name_list=feature_name_list,
        count_feature_name_list=count_feature_name_list,
        ds_info=ds_info
    )
    if curr_df is None:
        return None

    wake_time_idx, sleep_time_idx = get_sleep_start_and_end_times(
        pid=pid,
        day=day,
        dates_df=dates_df,
        sleep_summary_df=sleep_summary_df,
        day_list=day_list,
        ds_info=ds_info
    )

    if sleep_time_idx <= wake_time_idx:
        return None

    if sleep_time_idx > total_minutes:
        next_day = f'd{int(day[1:]) + 1:02d}'
        curr_awake_df = curr_df.iloc[wake_time_idx:].copy()
        if next_day not in day_list:
            return curr_awake_df
        next_path = os.path.join(feature_folder, f'{pid}-{next_day}-features.csv')
        next_df = load_feature_dataframe(
            file_path=next_path,
            feature_name_list=feature_name_list,
            count_feature_name_list=count_feature_name_list,
            ds_info=ds_info
        )
        if next_df is None:
            return curr_awake_df
        next_awake_df = next_df.iloc[:sleep_time_idx - total_minutes].copy()
        return pd.concat([curr_awake_df, next_awake_df], axis=0, ignore_index=True)

    return curr_df.iloc[wake_time_idx:sleep_time_idx].copy()


def get_sleep_dataframe(pid, wake_day, ds_info):
    file_path = os.path.join(ds_info['sleep-feat-path'], f'{pid}-{wake_day}-sleep-features.csv')
    return load_feature_dataframe(
        file_path=file_path,
        feature_name_list=ds_info['sleep-features-of-interest'],
        count_feature_name_list=ds_info['count-based-features'],
        ds_info=ds_info
    )


def has_minimum_valid_points(feature_df, min_valid_minutes):
    if feature_df is None or len(feature_df) == 0:
        return False
    valid_counts = feature_df.notna().sum(axis=0)
    return bool((valid_counts >= min_valid_minutes).all())


def append_feature_values(storage_dict, feature_df):
    for feature_name in feature_df.columns:
        valid_values = feature_df[feature_name].dropna().to_numpy()
        if valid_values.size > 0:
            storage_dict[feature_name].append(valid_values)


def compute_feature_stats(feature_storage, feature_name_list):
    mean_list = []
    std_list = []
    for feature_name in feature_name_list:
        value_chunks = feature_storage[feature_name]
        if len(value_chunks) == 0:
            raise ValueError(f'No valid values collected for feature: {feature_name}')
        values = np.concatenate(value_chunks, axis=0)
        mean_list.append(np.mean(values))
        std_list.append(np.std(values))
    return np.array(mean_list), np.array(std_list)


def append_sleep_stat_values(storage_dict, sleep_summary_df, pid, wake_day, ds_info):
    sleep_summary_df = sleep_summary_df[sleep_summary_df['start_time'] != 'error']
    sleep_stat_df = sleep_summary_df[
        (sleep_summary_df['PID'] == pid)
        & (sleep_summary_df['Wake Day'] == wake_day)
    ][SLEEP_STAT_FEATURE_NAMES].copy()
    sleep_stat_df = sleep_stat_df.dropna()
    if sleep_stat_df.empty:
        return False

    assert len(sleep_stat_df) == 1, f'Expected 1 sleep summary row for PID: {pid}, Day: {wake_day}, but got {len(sleep_stat_df)}'
    if 'start_time' in SLEEP_STAT_FEATURE_NAMES:
        sleep_stat_df['start_time'] = sleep_stat_df['start_time'].apply(
            lambda x: convert_to_minutes(x, fmt='YYYY/MM/DD HH:MM:SS')
        )
        if sleep_stat_df['start_time'].iloc[0] > ds_info['night-minutes']:
            sleep_stat_df['start_time'] = sleep_stat_df['start_time'] - ds_info['total-minutes']
    if 'end_time' in SLEEP_STAT_FEATURE_NAMES:
        sleep_stat_df['end_time'] = sleep_stat_df['end_time'].apply(
            lambda x: convert_to_minutes(x, fmt='YYYY/MM/DD HH:MM:SS')
        )
        assert sleep_stat_df['end_time'].iloc[0] < ds_info['night-minutes'], 'End time is not less than night minutes!'
    sleep_stat_df = sleep_stat_df.astype(float)
    sleep_stat_df.loc[:, 'duration'] = sleep_stat_df['duration'] / 3600.0
    append_feature_values(storage_dict, sleep_stat_df)
    return True


def compute_dg_stats():
    gender_mean = GENDER_BERNOULLI_P
    gender_std = np.sqrt(GENDER_BERNOULLI_P * (1.0 - GENDER_BERNOULLI_P))
    age_mean = (AGE_UNIFORM_MIN + AGE_UNIFORM_MAX) / 2.0
    age_std = (AGE_UNIFORM_MAX - AGE_UNIFORM_MIN) / np.sqrt(12.0)

    dg_mean = np.array([gender_mean, age_mean])
    dg_std = np.array([gender_std, age_std])
    return dg_mean, dg_std


def compute_time_stats():
    time_mean = np.array([(TIME_UNIFORM_MIN + TIME_UNIFORM_MAX) / 2.0])
    time_std = np.array([(TIME_UNIFORM_MAX - TIME_UNIFORM_MIN) / np.sqrt(12.0)])
    return time_mean, time_std


def main():
    non_sleep_feature_storage = {}
    sleep_feature_storage = {}
    sleep_stat_storage = {feature: [] for feature in SLEEP_STAT_FEATURE_NAMES}
    non_sleep_feature_names = None
    sleep_feature_names = None
    used_non_sleep_days_dict = {}
    used_sleep_days_dict = {}
    excluded_non_sleep_pids = []
    excluded_sleep_pids = []

    for dataset_country in DATASET_COUNTRIES:
        ds_info = build_ds_info(dataset_country=dataset_country)
        dates_df = pd.read_csv(ds_info[f'{dataset_country}-dates-path'])
        sleep_summary_df = pd.read_csv(ds_info['sleep-stat-path'])

        if non_sleep_feature_names is None:
            non_sleep_feature_names = ds_info['features-of-interest']
            non_sleep_feature_storage = {feature: [] for feature in non_sleep_feature_names}
        if sleep_feature_names is None:
            sleep_feature_names = ds_info['sleep-features-of-interest']
            sleep_feature_storage = {feature: [] for feature in sleep_feature_names}

        for pid in ds_info['pid-list']:
            valid_non_sleep_days = []
            valid_sleep_days = []

            for day in FIXED_DAYS:
                if day not in ds_info['day-list']:
                    continue
                non_sleep_df = get_non_sleep_dataframe(
                    pid=pid,
                    day=day,
                    ds_info=ds_info,
                    dates_df=dates_df,
                    sleep_summary_df=sleep_summary_df
                )
                if has_minimum_valid_points(non_sleep_df, MIN_VALID_MINUTES):
                    append_feature_values(non_sleep_feature_storage, non_sleep_df)
                    valid_non_sleep_days.append(day)

                sleep_df = get_sleep_dataframe(pid=pid, wake_day=day, ds_info=ds_info)
                if has_minimum_valid_points(sleep_df, MIN_VALID_MINUTES):
                    append_feature_values(sleep_feature_storage, sleep_df)
                    valid_sleep_days.append(day)

                append_sleep_stat_values(
                    storage_dict=sleep_stat_storage,
                    sleep_summary_df=sleep_summary_df,
                    pid=pid,
                    wake_day=day,
                    ds_info=ds_info
                )

            if valid_non_sleep_days:
                used_non_sleep_days_dict[pid] = valid_non_sleep_days
            else:
                excluded_non_sleep_pids.append(pid)

            if valid_sleep_days:
                used_sleep_days_dict[pid] = valid_sleep_days
            else:
                excluded_sleep_pids.append(pid)

    non_sleep_mean, non_sleep_std = compute_feature_stats(
        feature_storage=non_sleep_feature_storage,
        feature_name_list=non_sleep_feature_names
    )
    sleep_mean, sleep_std = compute_feature_stats(
        feature_storage=sleep_feature_storage,
        feature_name_list=sleep_feature_names
    )
    sleep_stat_mean, sleep_stat_std = compute_feature_stats(
        feature_storage=sleep_stat_storage,
        feature_name_list=SLEEP_STAT_FEATURE_NAMES
    )
    dg_mean, dg_std = compute_dg_stats()
    time_mean, time_std = compute_time_stats()

    non_sleep_pids = np.array(sorted(used_non_sleep_days_dict.keys()))
    non_sleep_days = np.empty(len(non_sleep_pids), dtype=object)
    for idx, pid in enumerate(non_sleep_pids):
        non_sleep_days[idx] = np.array([int(day[1:]) for day in used_non_sleep_days_dict[pid]], dtype=int)

    sleep_pids = np.array(sorted(used_sleep_days_dict.keys()))
    sleep_days = np.empty(len(sleep_pids), dtype=object)
    for idx, pid in enumerate(sleep_pids):
        sleep_days[idx] = np.array([int(day[1:]) for day in used_sleep_days_dict[pid]], dtype=int)

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    np.savez(
        SAVE_PATH,
        dataset_countries=np.array(DATASET_COUNTRIES),
        fixed_days=np.array([int(day[1:]) for day in FIXED_DAYS]),
        min_valid_minutes=np.array(MIN_VALID_MINUTES),
        num_selected_non_sleep_pids=np.array(len(non_sleep_pids)),
        num_selected_sleep_pids=np.array(len(sleep_pids)),
        non_sleep_feature_names=np.array(non_sleep_feature_names),
        non_sleep_mean=non_sleep_mean,
        non_sleep_std=non_sleep_std,
        sleep_feature_names=np.array(sleep_feature_names),
        sleep_mean=sleep_mean,
        sleep_std=sleep_std,
        sleep_stat_feature_names=np.array(SLEEP_STAT_FEATURE_NAMES),
        sleep_stat_mean=sleep_stat_mean,
        sleep_stat_std=sleep_stat_std,
        dg_feature_names=np.array(DG_FEATURE_NAMES),
        dg_mean=dg_mean,
        dg_std=dg_std,
        time_feature_names=np.array(TIME_FEATURE_NAMES),
        time_mean=time_mean,
        time_std=time_std,
        non_sleep_pids=non_sleep_pids,
        non_sleep_days=non_sleep_days,
        sleep_pids=sleep_pids,
        sleep_days=sleep_days,
        excluded_non_sleep_pids=np.array(sorted(excluded_non_sleep_pids)),
        excluded_sleep_pids=np.array(sorted(excluded_sleep_pids)),
    )

    print(f'Saved z-score parameters to: {SAVE_PATH}')
    print(f'Participants used for non-sleep stats: {len(non_sleep_pids)}')
    print(f'Participants used for sleep stats: {len(sleep_pids)}')
    print(f'Participants excluded from non-sleep stats: {len(excluded_non_sleep_pids)}')
    print(f'Participants excluded from sleep stats: {len(excluded_sleep_pids)}')


if __name__ == '__main__':
    main()
