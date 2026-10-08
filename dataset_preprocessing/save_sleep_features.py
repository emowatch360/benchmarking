# main.py
import os
import pandas as pd
from config import (
    SAVE_FOLDER as PROCESSED_DATA_FOLDER, 
    DATES_CSV, NUM_TIME_SAMPLES,
    SLEEP_SUMMARY_CSV, SLEEP_SAVE_FOLDER, dataset_country as country_in_config
)
from utils import compute_day_and_time_indices, convert_to_minutes
from data_cleaning import load_sleep_feature_data
from datetime import timedelta
max_days = 28
log_folder = 'log'
os.makedirs(log_folder, exist_ok=True)
log_file_path = os.path.join(log_folder, f'sleep_feature_log_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.txt')

def main(chosen_day, chosen_pid, sleep_time_dict):
    start_day = sleep_time_dict['Start Day']
    start_time = sleep_time_dict['Start Time']
    end_day = sleep_time_dict['End Day']
    end_time = sleep_time_dict['End Time']
    start_csv_path = os.path.join(PROCESSED_DATA_FOLDER, f'{chosen_pid}-{start_day}-features.csv')
    end_csv_path = os.path.join(PROCESSED_DATA_FOLDER, f'{chosen_pid}-{end_day}-features.csv')
    # check if the files exist
    if not os.path.exists(start_csv_path):
        with open(log_file_path, 'a') as log_file:
            log_file.write(f"Start CSV file not found: {start_csv_path}, skipping...\n")
        return
    if not os.path.exists(end_csv_path):
        with open(log_file_path, 'a') as log_file:
            log_file.write(f"End CSV file not found: {end_csv_path}, skipping...\n")
        return
    if start_day == end_day:
        feature_df = load_sleep_feature_data(read_path = start_csv_path,
                                               start_time_index = start_time,
                                               end_time_index = end_time,
                                               num_time_samples=NUM_TIME_SAMPLES)
        feature_df.insert(0, 'Day', start_day)
        assert len(feature_df) == (end_time-start_time+1)
    else:
        start_feature_df = load_sleep_feature_data(read_path = start_csv_path,
                                               start_time_index = start_time,
                                               end_time_index = NUM_TIME_SAMPLES-1, #until 23:59
                                               num_time_samples=NUM_TIME_SAMPLES)
        start_feature_df.insert(0, 'Day', start_day)
        end_feature_df = load_sleep_feature_data(read_path = end_csv_path,
                                               start_time_index = 0, #from 00:00
                                               end_time_index = end_time,
                                               num_time_samples=NUM_TIME_SAMPLES)
        end_feature_df.insert(0, 'Day', end_day)
        feature_df = pd.concat([start_feature_df, end_feature_df], ignore_index=True)
        assert len(feature_df) == (end_time+1+NUM_TIME_SAMPLES-start_time)
    # Ensure that the save directory exists.
    os.makedirs(SLEEP_SAVE_FOLDER, exist_ok=True)
    save_path = os.path.join(SLEEP_SAVE_FOLDER, f'{chosen_pid}-{chosen_day}-sleep-features.csv')
    feature_df.to_csv(save_path, index=False)
    with open(log_file_path, 'a') as log_file:
        log_file.write(f"Feature data saved to {save_path}\n")

if __name__ == '__main__':
    sleep_summary_df = pd.read_csv(SLEEP_SUMMARY_CSV)
    pid_list = sleep_summary_df['PID']
    day_list = sleep_summary_df['Wake Day']
    start_timestamp_list = sleep_summary_df['start_time']
    end_timestamp_list = sleep_summary_df['end_time']
    dates_df = pd.read_csv(DATES_CSV)
    dataset_country = 'jp'
    assert country_in_config.lower() == dataset_country.lower(), "Dataset country mismatch!"
    # valid_idx = 0
    for idx, (chosen_pid, chosen_day, start_timestamp, end_timestamp) in enumerate(zip(pid_list, day_list, start_timestamp_list, end_timestamp_list)):
        if dataset_country == 'jp':
            if 'MM' not in chosen_pid:
                continue
        elif dataset_country == 'sg':
            if 'pid' not in chosen_pid and 'oac' not in chosen_pid:
                continue
        with open(log_file_path, 'a') as log_file:
            log_file.write(f'Processing index {idx}...\n')
            log_file.write(f'Chosen PID: {chosen_pid}, Chosen Day: {chosen_day}\n')
        if chosen_day != chosen_day: # nan
            continue
        elif int(chosen_day[1:]) > max_days or int(chosen_day[1:])<1: # ignore day29,...
            continue
        if start_timestamp == 'error' or start_timestamp == '' or start_timestamp != start_timestamp or \
            end_timestamp == 'error' or end_timestamp == '' or end_timestamp != end_timestamp: # nan
            continue
        # identify first date str
        if dataset_country == 'jp':
            row = dates_df.loc[dates_df['pid'] == chosen_pid, 'start date']
            if row.empty:
                raise ValueError(f"No start date found for pid={chosen_pid} in {DATES_CSV}")
            first_date_str = row.iloc[0]
            start_day, start_time = compute_day_and_time_indices(start_timestamp, first_date_str=first_date_str, fmt='YYYY-MM-DD HH:MM:SS')
            end_day, end_time = compute_day_and_time_indices(end_timestamp, first_date_str=first_date_str, fmt='YYYY-MM-DD HH:MM:SS')
            if int(start_day[1:]) < 1 or int(end_day[1:]) > max_days: # start_day can be d00, end_day can be d29
                with open(log_file_path, 'a') as log_file:
                    log_file.write(f'Invalid start day/end day: {start_day}/{end_day} for pid={chosen_pid}, skipping...\n')
                continue
        elif dataset_country == 'sg':
            end_time = convert_to_minutes(end_timestamp, fmt='YYYY-MM-DD HH:MM:SS')
            start_time = convert_to_minutes(start_timestamp, fmt='YYYY-MM-DD HH:MM:SS')
            # rows are chosen_pid, and columns as chosen_day
            # convert the date str from 'YYYY/MM/DD' to 'YYYY-MM-DD'
            row = dates_df.loc[dates_df['pid'] == chosen_pid, chosen_day]
            if row.empty:
                raise ValueError(f"No date found for pid={chosen_pid}, day={chosen_day} in {DATES_CSV}")
            # check that end_timestamp is the same as chosen_day date
            wake_date_str = row.iloc[0]
            assert end_timestamp.startswith(wake_date_str.replace('/', '-')), 'end timestamp date does not match wake day date'
            end_day = chosen_day
            if int(end_day[1:]) > max_days:
                with open(log_file_path, 'a') as log_file:
                    log_file.write(f'Invalid end day: {end_day} for pid={chosen_pid}, skipping...\n')
                continue
            # compute start_day
            if start_timestamp.startswith(wake_date_str):
                start_day = chosen_day
            else:
                # previous day
                start_day = f'd{int(chosen_day[1:])-1:02d}'
                # check that start_day is valid
                # in this dataset, the previous day may not always be the chronological previous day
                if int(start_day[1:]) < 1: # start_day can be d00, end_day can be d29
                    with open(log_file_path, 'a') as log_file:
                        log_file.write(f'Invalid start day: {start_day} for pid={chosen_pid}, skipping...\n')
                    continue
                if dates_df.loc[dates_df['pid'] == chosen_pid, start_day].iloc[0] != (pd.to_datetime(wake_date_str) - timedelta(days=1)).strftime('%Y/%m/%d'):
                    with open(log_file_path, 'a') as log_file:
                        log_file.write(f'Start day {start_day} date does not match expected previous date for pid={chosen_pid}, skipping...\n')
        assert end_day == chosen_day
        sleep_time_dict = {
            'Start Day': start_day,
            'Start Time': start_time,
            'End Day': end_day,
            'End Time': end_time
        }
        with open(log_file_path, 'a') as log_file:
            log_file.write(f'Processing PID: {chosen_pid}, Day: {chosen_day}, Sleep Time Info: {sleep_time_dict}\n')
        main(chosen_day, chosen_pid, sleep_time_dict)