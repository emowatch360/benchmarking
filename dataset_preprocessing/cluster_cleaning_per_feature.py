import os
import pandas as pd
from datetime import datetime, timedelta
from utils import compute_day_id #type:ignore

feature_list = ['MeanNN', 'RMSSD', 'SDSD', 'CVSD',
       'MedianNN', 'MadNN', 'IQRNN', 'Prc20NN', 'Prc80NN', 'pNN50', 'pNN20',
       'MinNN', 'MaxNN', 'HTI', 'HF', 'VHF', 'TP', 'LnHF', 'SD1', 'S', 'CVI',
       'PIP', 'IALS', 'PSS', 'PAS', 'SD1d', 'SD1a', 'MFDFA_alpha2_Width',
       'MFDFA_alpha2_Delta', 'MFDFA_alpha2_Asymmetry',
       'MFDFA_alpha2_Fluctuation', 'MFDFA_alpha2_Increment', 'ApEn', 'Spo2',
       'Total Energy', 'Zero Crossing Count', 'Time Above Threshold',
       'Respiration (breaths/min)']
# first_date_str = '2024-11-04'

# Load input CSV
data_folder = '/data/emowatch/emowatch/'
input_path = os.path.join(data_folder, 'sleep_summary_cluster10d.csv')
output_folder = os.path.join(data_folder, 'sleep_clusters')
df = pd.read_csv(input_path)
# remove rows when 'end' feature is empty or 'error
df = df[(df['end_time'].notna()) & (df['end_time'] != '') & (df['end_time'] != 'error')]
# Compute 'Next Day' for each row
# df['Next Day'] = df['end'].apply(lambda x: compute_day_id(x.split(' ')[0], first_date_str))
for feature in feature_list:
    print(f'Feature:{feature}')
    df_output = df[['PID', 'Wake Day', feature]].copy()
    df_output.dropna(inplace=True)
    df_output[feature] = df_output[feature].astype(int)
    df_output = df_output.rename(columns={feature: 'Cluster'})
    # Save to new CSV
    output_path = os.path.join(output_folder, f"sleep_cluster_{feature.split('(')[0]}.csv")
    df_output.to_csv(output_path, index=False)