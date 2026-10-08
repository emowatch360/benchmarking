import numpy as np
import pandas as pd
import os
from datetime import datetime
from utils import convert_to_minutes
from config import (ROOT_FOLDER)

dataset_country = 'SG'
data_path = os.path.join(ROOT_FOLDER, f'rawdata_{dataset_country.lower()}')
questionnaire_file = 'Questionnaire-sg_2025-10-30.csv'
input_path = os.path.join(data_path, questionnaire_file)
output_path = os.path.join(ROOT_FOLDER, f'processed_labels_{dataset_country.lower()}_v1.csv')

#'PID', 'Day', 'Session', 'Timestamp', 
output_cols = ['valence_current', 'valence_ideal', 'arousal_current', 'arousal_ideal',
				'stress_level', 'attention_level', 'emotion_duration', 
				'drink_craving', 'drink_category', 'drink_others',
				'sleep_quality', 'body_condition']

# read through each row
df = pd.read_csv(input_path, header=0)
new_df = pd.DataFrame()
new_df['PID'] = df['pid']
date_list = [datetime.strptime(date, '%Y/%m/%d').strftime('%d-%m-%Y') for date in df['Date']]
# concatenate day_id and date -> day_id + '_' + modified date with '/' replaced by '-'
new_df['Day'] = [f"{df['day_id'].iloc[i]}_{date_list[i]}" for i in range(len(df))]
new_df['Time Index'] = [convert_to_minutes(x.split(' ')[1], 'HH:MM') for x in df['RecordedDate']]
new_df['Timestamp'] = [x.split(' ')[1] for x in df['RecordedDate']]
new_df['Session'] = df['Timing']
new_df['SurveySeconds'] = df['Duration (in seconds)']
for col in output_cols:
	new_df[col] = df[col]
# new_df.to_csv(output_path, index=False)