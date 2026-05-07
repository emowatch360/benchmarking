# go through the whole folder structure of hrv_sleep and get the PID, Wake Day, start_time and end_time of sleep
import os
import pandas as pd
from utils import convert_to_minutes

hrv_folder = '/data/emowatch/emowatch/hrv_indices_sleep'
sleep_csvs = ['sleep_summary_c1_cleaned.csv', 'sleep_summary_c2ins_cleaned.csv', 'sleep_summary_c2com_cleaned.csv']
save_path = '/data/emowatch/emowatch/sleep_summary_all.csv'

dfs = []
for sleep_csv in sleep_csvs:
    sleep_csv_path = os.path.join(hrv_folder, sleep_csv)
    df = pd.read_csv(sleep_csv_path)
    dfs.append(df)
df_all = pd.concat(dfs, ignore_index=True)
# rename columns
# change 'day_id' to 'Wake Day'
df_all = df_all.rename(columns={'day_id': 'Wake Day'})
df_all = df_all.rename(columns={'User First Name': 'PID'})
# retain only some rows
df_all = df_all[['PID', 'Wake Day', 'start_time', 'end_time', 'duration', 'Rem_per', 'Deep_per', 'Light_per', 'Awake_per']]
df_all.to_csv(save_path, index=False)

# all_pids = os.listdir(hrv_folder_path)
# stat_list = []
# for pid in all_pids:
#     pid_folder = os.path.join(hrv_folder_path, pid)
#     if not os.path.isdir(pid_folder):
#         continue
#     if not ('MM' in pid or 'pid' in pid or 'c2com' in pid):
#         continue
#     all_days = os.listdir(pid_folder)
#     for day in all_days:
#         day_folder = os.path.join(pid_folder, day)
#         if not os.path.isdir(day_folder):
#             continue
#         hrv_file_path = os.path.join(day_folder, 'hrv_sleep.csv')
#         if not os.path.isfile(hrv_file_path):
#             continue
#         hrv_df = pd.read_csv(hrv_file_path)
#         if hrv_df.empty:
#             continue
#         start_time = hrv_df['Start Time'].iloc[0]
#         end_time = hrv_df['Start Time'].iloc[-1]
#         start_time = start_time.replace('-', '/')
#         end_time = end_time.replace('-', '/')
#         # duration of sleep
#         start_min_idx = convert_to_minutes(start_time, fmt='YYYY/MM/DD HH:MM:SS')
#         end_min_idx = convert_to_minutes(end_time, fmt='YYYY/MM/DD HH:MM:SS')
#         # same date
#         if start_time.split(' ')[0] == end_time.split(' ')[0]:
#             duration = end_min_idx - start_min_idx + 1
#         else:
#             duration = 1440 - start_min_idx + end_min_idx + 1
#         # save duration in seconds
#         stat_list.append({'PID': pid, 'Wake Day': day, 'start_time': start_time, 'end_time': end_time,
#                             'duration': duration*60, 'Rem_per': 25.0, 'Deep_per':25.0, 'Light_per':25.0, 'Awake_per':25.0}) # default values
# df = pd.DataFrame(stat_list)
# df.to_csv(save_path, index=False)
