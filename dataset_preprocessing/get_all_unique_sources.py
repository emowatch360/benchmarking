import os
import pandas as pd

file_names = ['skin-temperature.csv', 'actigraphy.csv', 'heart-rate.csv', 'motion-intensity.csv',
              'step-log.csv', 'respiration.csv', 'stress.csv', 'spo2-logging.csv', 'spo2.csv']
dir_path = '/data/emowatch/emowatch/rawdata_jp'
source_dict = {}
for file_name in file_names:
    source_dict[file_name] = set()

for sub_folder in os.listdir(dir_path):
    folder_path = os.path.join(dir_path, sub_folder)
    if os.path.isdir(folder_path):
        for file_name in file_names:
            file_path = os.path.join(folder_path, file_name)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                if 'Source' in df.columns:
                    sources = df['Source'].unique()
                    source_dict[file_name].update(sources)

for file_name, sources in source_dict.items():
    print(f"File: {file_name}")
    print(f"Unique Sources: {sources}\n")
    print('-'*100)