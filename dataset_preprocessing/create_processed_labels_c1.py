import numpy as np
import pandas as pd
import os
from datetime import datetime
from utils import convert_to_minutes
from config import (ROOT_FOLDER)

dataset_country = 'C1'
data_path = os.path.join(ROOT_FOLDER, f'rawdata_{dataset_country.lower()}')
questionnaire_file = 'Questionnaire_09-01-2025.csv'
input_path = os.path.join(data_path, questionnaire_file)
output_path = os.path.join(data_path, f'labels_{dataset_country.lower()}_v1.csv')

#'PID', 'Day', 'Session', 'Timestamp', 
output_cols = ['valence_current', 'valence_ideal', 'arousal_current', 'arousal_ideal',
				'relaxed_current', 'relaxed_ideal', 'energetic_current', 'energetic_ideal',
				'tired_current', 'tired_ideal', 'stressed_current', 'stressed_ideal',
				'work_ornot', 'work_place', 'work_concentration', 'work_performance', 
				'sleep_quality', 'body_condition', 'panas_positive', 'panas_negative']
first_date = '2024/11/04' # applicable to c1 dataset only

def modify_date(curr_datetime_str, first_date_str):
	curr_date_str = curr_datetime_str.split(' ')[0]
	curr_date = datetime.strptime(curr_date_str, '%Y/%m/%d')
	first_date = datetime.strptime(first_date_str, '%Y/%m/%d')
	date_diff = (curr_date-first_date).days+1
	modified_date = 'd' + f'0{date_diff}'[-2:] + '_' + curr_date.strftime("%d-%m-%Y")
	return modified_date

# read through each row
df = pd.read_csv(input_path, skiprows=1, header=0)
new_df = pd.DataFrame()
new_df['PID'] = df['Subject_id']
new_df['Day'] = [modify_date(x, first_date) for x in df['FinishDate']]
new_df['Time Index'] = [convert_to_minutes(x.split(' ')[1], 'HH:MM') for x in df['FinishDate']]
new_df['Timestamp'] = [x.split(' ')[1] for x in df['FinishDate']]
new_df['Session'] = df['Timing']
new_df['SurveySeconds'] = df['SurveySeconds']
for col in output_cols:
	new_df[col] = df.filter(like=col).bfill(axis=1).iloc[:,0]
# new_df.to_csv(output_path, index=False)