# main.py
import os
import numpy as np
import pandas as pd
from config import (
    FEATURE_FOLDER, HRV_FOLDER, SAVE_FOLDER, DATES_CSV, dataset_country as country_in_config,
    NUM_TIME_SAMPLES
)
from utils import compute_day_str, convert_to_minutes
STEP_DICT = {'feature': 'Steps', 'csv_name': 'intraday'}

# create a log file with current timestamp in the log/ folder
log_folder = 'log'
os.makedirs(log_folder, exist_ok=True)
log_file_path = os.path.join(log_folder, f'intraday_log_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.txt')

def load_step_data(chosen_pid_folder, date_str, step_dict,
                     num_time_samples, time_interval=15, log_file_path=None):  
    """
    Load and process Step data.
    Returns an array of shape (num_time_samples, ).
    """
    step_data = np.full((num_time_samples,), np.nan)
    step_file_path = os.path.join(chosen_pid_folder, step_dict['csv_name'] + '.csv')
    step_df = pd.read_csv(step_file_path)
    if step_df.empty:
        raise ValueError(f"Step DataFrame from {step_file_path} is empty.")
    chosen_df = step_df[step_df['Start Time (Local)'].str.contains(date_str)]
    if chosen_df.empty:
        with open(log_file_path, 'a') as log_file:
            log_file.write(f"{chosen_pid_folder}: No step data for date {date_str} in file {step_file_path}\n")
        return step_data
    # Filter dataframe by timestamp date_str, and convert to minutes
    start_minutes = np.array([convert_to_minutes(t) for t in chosen_df['Start Time (Local)']])
    end_minutes = np.array([convert_to_minutes(t) for t in chosen_df['End Time (Local)']])
    # for each row, fill in the step_data array from start_minutes to end_minutes with the step count
    for i, idx in enumerate(chosen_df.index):
        s_min = start_minutes[i]
        e_min = end_minutes[i]
        if e_min < s_min:
            e_min = num_time_samples # correct end time to 00:00 of next day
        if not (e_min <= s_min + time_interval):
            with open(log_file_path, 'a') as log_file:
                log_file.write(f"{chosen_pid_folder}: "
                               f"Warning: Index: {idx}, End minute {e_min} exceeds start minute {s_min} by more than {time_interval}!\n")
            s_min = max(0, e_min - time_interval)  #  correcting start time
            with open(log_file_path, 'a') as log_file:
                log_file.write(f"{chosen_pid_folder}: "
                               f"Corrected Start minute to {s_min}\n")
        # ensure that step data in the range is not already filled
        if not np.all(np.isnan(step_data[s_min:e_min])):
            with open(log_file_path, 'a') as log_file:
                log_file.write(f"{chosen_pid_folder}: "
                               f"Warning: Index: {idx}, Some step data from {s_min} to {e_min} already filled!\n")
        step_count = chosen_df.loc[idx, step_dict['feature']]
        assert (s_min < num_time_samples) and (e_min <= num_time_samples), "One or more timestamp indices are out-of-range!"
        # divide the steps evenly over the minutes
        if s_min == e_min:
            with open(log_file_path, 'a') as log_file:
                log_file.write(f"{chosen_pid_folder}: "
                               f"Warning: Index: {idx}, Start minute {s_min} equals End minute {e_min}!\n")
            raise ValueError("Start minute equals End minute!")
        update_step = step_count // (e_min - s_min)
        update_step_vec = np.full((e_min - s_min,), update_step)
        # update step_data by adding to existing values (in case of overlapping intervals) while accounting for nan
        step_data[s_min:e_min] = np.nansum(np.vstack((step_data[s_min:e_min], update_step_vec)), axis=0)
        step_data[e_min-1] += step_count % (e_min - s_min)  # add remainder to last minute
    return step_data

def main(chosen_day, chosen_pid, dataset_country, step_data_min=0, step_data_max=300):
    # Find the folder corresponding to the chosen day
    pid_folder = os.path.join(FEATURE_FOLDER, chosen_pid)
    # read DATES_CSV, get the value of the 'start date' column for the 'pid' of chosen_pid
    dates_df = pd.read_csv(DATES_CSV)
    if dataset_country == 'C1':
        row = dates_df.loc[dates_df['pid'] == chosen_pid, 'start date']
        if row.empty:
            raise ValueError(f"No start date found for pid={chosen_pid} in {DATES_CSV}")
        first_date_str = row.iloc[0]
        date_str = compute_day_str(chosen_day, first_date_str)
    elif dataset_country == 'C2':
        # rows are chosen_pid, and columns as chosen_day
        # convert the date str from 'YYYY/MM/DD' to 'YYYY-MM-DD'
        row = dates_df.loc[dates_df['pid'] == chosen_pid, chosen_day]
        if row.empty:
            raise ValueError(f"No date found for pid={chosen_pid}, day={chosen_day} in {DATES_CSV}")
        date_str = row.iloc[0].replace('/', '-')
    # Load wearable feature data
    step_data = load_step_data(pid_folder, date_str, STEP_DICT, NUM_TIME_SAMPLES,
                                     log_file_path=log_file_path)
    # Ensure that the save directory exists.
    save_path = os.path.join(SAVE_FOLDER, f'{chosen_pid}-{chosen_day}-features.csv')
    df = pd.read_csv(save_path)
    if df.empty:
        raise ValueError(f"Feature DataFrame from {save_path} is empty.")
    # check that step count and total count columns are empty
    assert df['Step Count'].isnull().all(), f"Step Count column in {save_path} is not empty."
    assert df['Total Count'].isnull().all(), f"Total Count column in {save_path} is not empty."
    if np.all(np.isnan(step_data)):
        return
    # Sanity check: ensure per-minute step count is within [0, 300]
    invalid_mask = (step_data < step_data_min) | (step_data > step_data_max) # based on config
    if np.any(invalid_mask):
        with open(log_file_path, 'a') as log_file:
            log_file.write(
                f"{chosen_pid}, {chosen_day}: Found {np.sum(invalid_mask)} step values outside [0, 300]. Setting them to NaN.\n"
            )
        step_data[invalid_mask] = np.nan
    df['Step Count'] = step_data
    df['Total Count'] = np.nancumsum(step_data) # ignore missing values for now
    df.to_csv(save_path, index=False)
    # print(f"Feature data saved to {save_path}")
    # change print statements to write to log file
    with open(log_file_path, 'a') as log_file:
        log_file.write(f"Feature data saved to {save_path}\n")

if __name__ == '__main__':
    dataset_country = 'C2' # set the corresponding country in config.py too!
    assert dataset_country.lower() == country_in_config.lower(), f"Dataset country mismatch!"
    pid_list = os.listdir(HRV_FOLDER)
    # country - C1
    if dataset_country == 'C1':
        # only retain pids that start with 'MM'
        pid_list = [pid for pid in pid_list if pid.startswith('MM')]
    elif dataset_country == 'C2':
        # only retain pids that start with 'c2com' or 'pid'
        pid_list = [pid for pid in pid_list if pid.startswith('c2com') or pid.startswith('pid')] 
    days_list = [f'd{idx:02d}' for idx in range(1, 29)]

    pid_list_without_step_log = []
    for pid in pid_list:
        fpath = os.path.join(FEATURE_FOLDER, pid, 'step-log.csv')
        df = pd.read_csv(fpath)
        if df.empty: # use intraday only if step-log is empty
            pid_list_without_step_log.append(pid)
    pid_list = pid_list_without_step_log

    with open(log_file_path, 'a') as log_file:
        log_file.write('='*100 + '\n')
        log_file.write(f"Total PIDs to process: {len(pid_list)}\n")
        log_file.write(f"{pid_list}\n")
        log_file.write(f"Total days to process: {len(days_list)}\n")
        log_file.write(f"{days_list}\n")
        log_file.write('='*100 + '\n')

    for chosen_pid in pid_list:
        for chosen_day in days_list:
            with open(log_file_path, 'a') as log_file:
                log_file.write('='*100 + '\n')
            main(chosen_day, chosen_pid, dataset_country, step_data_min=0, step_data_max=300)