# main.py
import os
import numpy as np
import pandas as pd
from config import (
    FEATURE_FOLDER, HRV_FOLDER, SAVE_FOLDER, DATES_CSV, HRV_CSV, 
    FEATURE_DICT_LIST, HRV_NAME_LIST, 
    ALL_FEATURE_NAME_LIST, NUM_TIME_SAMPLES, dataset_country as country_in_config
)
from utils import DELTA, compute_day_str
from data_cleaning import load_feature_data, load_hrv_data

# create a log file with current timestamp in the log/ folder
log_folder = 'log'
os.makedirs(log_folder, exist_ok=True)
log_file_path = os.path.join(log_folder, f'feature_log_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.txt')

def main(chosen_day, chosen_pid, dataset_country):
    if country_in_config.lower() != dataset_country.lower():
        raise ValueError(f"Dataset country in config.py ({country_in_config}) does not match the argument ({dataset_country})")
    # Find the folder corresponding to the chosen day
    pid_folder = os.path.join(FEATURE_FOLDER, chosen_pid)
    assert os.path.exists(pid_folder), f"PID folder not found: {pid_folder}"
    # compute date string
    # read DATES_CSV, get the value of the 'start date' column for the 'pid' of chosen_pid
    dates_df = pd.read_csv(DATES_CSV)
    if dataset_country == 'JP':
        row = dates_df.loc[dates_df['pid'] == chosen_pid, 'start date']
        if row.empty:
            raise ValueError(f"No start date found for pid={chosen_pid} in {DATES_CSV}")
        first_date_str = row.iloc[0]
        date_str = compute_day_str(chosen_day, first_date_str)
    elif dataset_country == 'SG':
        # rows are chosen_pid, and columns as chosen_day
        # convert the date str from 'YYYY/MM/DD' to 'YYYY-MM-DD'
        row = dates_df.loc[dates_df['pid'] == chosen_pid, chosen_day]
        if row.empty:
            raise ValueError(f"No date found for pid={chosen_pid}, day={chosen_day} in {DATES_CSV}")
        date_str = row.iloc[0].replace('/', '-')
    # Load wearable feature data
    feature_data = load_feature_data(pid_folder, date_str, FEATURE_DICT_LIST, NUM_TIME_SAMPLES,
                                     log_file_path=log_file_path)
    # Load HRV data
    hrv_data = load_hrv_data(HRV_FOLDER, chosen_pid, chosen_day,
                             HRV_NAME_LIST, NUM_TIME_SAMPLES, date_str, 
                             hrv_fname=HRV_CSV,
                             log_file_path=log_file_path)
    # Combine feature data and HRV data horizontally.
    combined_mat = np.hstack((feature_data, hrv_data))
    # Create a DataFrame to save the data.
    save_df = pd.DataFrame(combined_mat, columns=ALL_FEATURE_NAME_LIST)
    save_df.insert(0, 'Time Index', np.arange(NUM_TIME_SAMPLES))
    save_df.insert(1, 'Timestamp', [f'{t // 60:02d}:{t % 60:02d}' for t in range(NUM_TIME_SAMPLES)])
    # Ensure that the save directory exists.
    os.makedirs(SAVE_FOLDER, exist_ok=True)
    save_path = os.path.join(SAVE_FOLDER, f'{chosen_pid}-{chosen_day}-features.csv')
    save_df.to_csv(save_path, index=False)
    # print(f"Feature data saved to {save_path}")
    # change print statements to write to log file
    with open(log_file_path, 'a') as log_file:
        log_file.write(f"Feature data saved to {save_path}\n")

if __name__ == '__main__':
    dataset_country = 'SG' # set the corresponding country in config.py too!
    pid_list = os.listdir(HRV_FOLDER)
    # country - JP
    if dataset_country == 'JP':
        # only retain pids that start with 'MM'
        pid_list = [pid for pid in pid_list if pid.startswith('MM')]
    elif dataset_country == 'SG':
        # only retain pids that start with 'oac' or 'pid'
        pid_list = [pid for pid in pid_list if pid.startswith('oac') or pid.startswith('pid')] 
    days_list = [f'd{idx:02d}' for idx in range(1, 29)]

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
                log_file.write('='*100+'\n')
            main(chosen_day, chosen_pid, dataset_country)