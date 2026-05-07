import os

dataset_country = 'c1' #c1 or c2
# --- Directories ---
ROOT_FOLDER = '/data/emowatch/emowatch'  # e.g., the parent folder (emowatch folder)
FEATURE_FOLDER = os.path.join(ROOT_FOLDER, f'rawdata_{dataset_country}')
HRV_FOLDER = os.path.join(ROOT_FOLDER, 'hrv_indices')
DATES_CSV = os.path.join(ROOT_FOLDER, f'{dataset_country}_start_end_dates.csv')
SLEEP_SUMMARY_CSV = os.path.join(ROOT_FOLDER, 'sleep_summary_all.csv') #_cluster10d.csv')
SAVE_FOLDER = os.path.join(ROOT_FOLDER, 'processed_data_all_filtered_temp') # adding temp to prevent overwrite of previous results
SLEEP_SAVE_FOLDER = os.path.join(ROOT_FOLDER, 'processed_sleep_data_all_filtered_temp') # adding temp to prevent overwrite of previous results
HRV_CSV = 'hrv_indices.csv'  # or 'hrv_indices_ov70.csv' # which file to read within the hrv_indices folder
''' old folder structure '''
# FEATURE_FOLDER = os.path.join(ROOT_FOLDER, 'Wearables_09-01-2025')
# HRV_FOLDER = os.path.join(ROOT_FOLDER, 'HRV-indices-Raw_2025-03-04')
# SLEEP_SUMMARY_CSV = os.path.join(ROOT_FOLDER, 'sleep_summary_cluster10d.csv')
# SAVE_FOLDER = os.path.join(ROOT_FOLDER, 'processed_data')
# SLEEP_SAVE_FOLDER = os.path.join(ROOT_FOLDER, 'processed_sleep_data')

# --- Dataset Settings ---
# FIRST_DATE_STR = '2024/11/04'
NUM_TIME_SAMPLES = 1440
MISSING_MARKER = -200

# --- Feature definitions ---
FEATURE_DICT_LIST = [
    {'feature': 'Respiration (breaths/min)', 'csv_list': ['respiration', 'respiration','respiration', 'respiration','respiration'], 'source_list': ['HUB_LOGGING', 'FIT_FILE', 'HUB-LOG', 'HUB', 'FIT_FILE_LOGGING'], 'ign_cond': '<6&>60'},
    {'feature': 'Heart Rate (bpm)', 'csv_list': ['heart-rate', 'heart-rate','heart-rate', 'heart-rate','heart-rate', 'heart-rate'], 'source_list': ['HUB_LOGGING', 'HUB-HR', 'FIT_FILE', 'HUB-LOG', 'HUB', 'FIT_FILE_LOGGING'], 'ign_cond': '<25&>220'}, # changed it from <=0
    {'feature': 'Spo2', 'csv_list': ['spo2', 'spo2-logging'], 'source_list': ['', ''], 'ign_cond': '<=0&>100'},
    {'feature': 'Temperature (celsius)', 'csv_list': ['skin-temperature','skin-temperature','skin-temperature'], 'source_list': ['HUB_LOGGING', 'HUB', 'FIT_FILE_LOGGING'], 'ign_cond': ''},
    {'feature': 'Stress Level Value', 'csv_list': ['stress', 'stress', 'stress', 'stress', 'stress', 'stress'], 'source_list': ['HUB_LOGGING', 'FIT_FILE', 'HUB-LOG', 'HUB-STRESS', 'HUB', 'FIT_FILE_LOGGING'], 'ign_cond': '<=0&>100'},
    {'feature': 'Body Battery (%)', 'csv_list': ['stress', 'stress', 'stress', 'stress', 'stress', 'stress'], 'source_list': ['HUB_LOGGING', 'FIT_FILE', 'HUB-LOG', 'HUB-STRESS', 'HUB', 'FIT_FILE_LOGGING'], 'ign_cond': '<=0&>100'},
    {'feature': 'Motion Intensity', 'csv_list': ['motion-intensity'], 'source_list': [''], 'ign_cond': '<0&>7'},
    {'feature': 'Step Count', 'csv_list': ['step-log'], 'source_list': [''], 'ign_cond': '<0&>300'},
    {'feature': 'Total Count', 'csv_list': ['step-log'], 'source_list': [''], 'ign_cond': ''},
    {'feature': 'Zero Crossing Count', 'csv_list': ['actigraphy'], 'source_list': [''], 'ign_cond': ''},
    {'feature': 'Total Energy', 'csv_list': ['actigraphy'], 'source_list': [''], 'ign_cond': '<=0'},
    {'feature': 'Time Above Threshold', 'csv_list': ['actigraphy'], 'source_list': [''], 'ign_cond': ''}
]

HRV_NAME_LIST = [
    # 'Seconds_BBI','Seconds_stamp','High_ratio','artifacts',
    'MeanNN','SDNN',
    'RMSSD','SDSD','CVNN','CVSD','MedianNN','MadNN','MCVNN','IQRNN','SDRMSSD','Prc20NN',
    'Prc80NN','pNN50','pNN20','MinNN','MaxNN','HTI','TINN','VLF','LF','HF','VHF','TP','LFHF',
    'LFn','HFn','LnHF','SD1','SD2','SD1SD2','S','CSI','CVI','CSI_Modified','PIP','IALS','PSS','PAS',
    'GI','SI','AI','PI','C1d','C1a','SD1d','SD1a','C2d','C2a','SD2d','SD2a','Cd','Ca','SDNNd','SDNNa',
    #'DFA_alpha1','MFDFA_alpha1_Width','MFDFA_alpha1_Peak','MFDFA_alpha1_Mean',
    #'MFDFA_alpha1_Max','MFDFA_alpha1_Delta','MFDFA_alpha1_Asymmetry',
    #'MFDFA_alpha1_Fluctuation','MFDFA_alpha1_Increment','DFA_alpha2',
    #'MFDFA_alpha2_Width','MFDFA_alpha2_Peak','MFDFA_alpha2_Mean',
    #'MFDFA_alpha2_Max','MFDFA_alpha2_Delta','MFDFA_alpha2_Asymmetry',
    #'MFDFA_alpha2_Fluctuation','MFDFA_alpha2_Increment',
    'ApEn','SampEn','ShanEn',
    'FuzzyEn','MSEn','CMSEn','RCMSEn','CD','HFD','KFD','LZC'
]

# Derived lists for use later
FEATURE_NAME_LIST = [f['feature'] for f in FEATURE_DICT_LIST]
ALL_FEATURE_NAME_LIST = FEATURE_NAME_LIST + HRV_NAME_LIST

# FEATURE_DICT_LIST = [
#     {'feature': 'Respiration (breaths/min)', 'csv_list': ['respiration', 'respiration','respiration', 'respiration','respiration'], 'source_list': ['HUB_LOGGING', 'FIT_FILE', 'HUB-LOG', 'HUB', 'FIT_FILE_LOGGING'], 'ign_cond': '<=0'},
#     {'feature': 'Heart Rate (bpm)', 'csv_list': ['heart-rate', 'heart-rate','heart-rate', 'heart-rate','heart-rate', 'heart-rate'], 'source_list': ['HUB_LOGGING', 'HUB-HR', 'FIT_FILE', 'HUB-LOG', 'HUB', 'FIT_FILE_LOGGING'], 'ign_cond': '<=25'}, # changed it from <=0
#     {'feature': 'Spo2', 'csv_list': ['spo2', 'spo2-logging'], 'source_list': ['', ''], 'ign_cond': '<=0&>100'},
#     {'feature': 'Temperature (celsius)', 'csv_list': ['skin-temperature','skin-temperature','skin-temperature'], 'source_list': ['HUB_LOGGING', 'HUB', 'FIT_FILE_LOGGING'], 'ign_cond': ''},
#     {'feature': 'Stress Level Value', 'csv_list': ['stress', 'stress', 'stress', 'stress', 'stress', 'stress'], 'source_list': ['HUB_LOGGING', 'FIT_FILE', 'HUB-LOG', 'HUB-STRESS', 'HUB', 'FIT_FILE_LOGGING'], 'ign_cond': '<=0&>100'},
#     {'feature': 'Body Battery (%)', 'csv_list': ['stress', 'stress', 'stress', 'stress', 'stress', 'stress'], 'source_list': ['HUB_LOGGING', 'FIT_FILE', 'HUB-LOG', 'HUB-STRESS', 'HUB', 'FIT_FILE_LOGGING'], 'ign_cond': '<=0&>100'},
#     {'feature': 'Motion Intensity', 'csv_list': ['motion-intensity'], 'source_list': [''], 'ign_cond': '<0&>7'},
#     {'feature': 'Step Count', 'csv_list': ['step-log'], 'source_list': [''], 'ign_cond': ''},
#     {'feature': 'Total Count', 'csv_list': ['step-log'], 'source_list': [''], 'ign_cond': ''},
#     {'feature': 'Zero Crossing Count', 'csv_list': ['actigraphy'], 'source_list': [''], 'ign_cond': ''},
#     {'feature': 'Total Energy', 'csv_list': ['actigraphy'], 'source_list': [''], 'ign_cond': '<=0'},
#     {'feature': 'Time Above Threshold', 'csv_list': ['actigraphy'], 'source_list': [''], 'ign_cond': ''}
# ]

# new features selected -- based on raw BBI
# based on pearson correlation : 49 features
# SELECTED_FEATURES = ['Respiration (breaths/min)', 'Heart Rate (bpm)', 'Spo2',
#        'Temperature (celsius)', 'Body Battery (%)', 'Motion Intensity',
#        'Step Count', 'Total Count', 'Zero Crossing Count', 'Total Energy',
#        'Seconds_BBI', 'High_ratio', 'artifacts', 'SDNN', 'RMSSD', 'SDRMSSD',
#        'TINN', 'VLF', 'LF', 'VHF', 'LFHF', 'LFn', 'HFn', 'PIP', 'PAS', 'GI',
#        'PI', 'C1d', 'C2d', 'DFA_alpha1', 'MFDFA_alpha1_Width',
#        'MFDFA_alpha1_Peak', 'MFDFA_alpha1_Mean', 'MFDFA_alpha1_Max',
#        'MFDFA_alpha1_Asymmetry', 'MFDFA_alpha1_Fluctuation', 'DFA_alpha2',
#        'MFDFA_alpha2_Width', 'MFDFA_alpha2_Peak', 'MFDFA_alpha2_Mean',
#        'MFDFA_alpha2_Max', 'MFDFA_alpha2_Delta', 'MFDFA_alpha2_Asymmetry',
#        'MFDFA_alpha2_Fluctuation', 'ApEn', 'ShanEn', 'MSEn', 'CMSEn',
#        'RCMSEn']

# # based on spearman correlation : 48 features
# SELECTED_FEATURES = ['Respiration (breaths/min)', 'Heart Rate (bpm)', 'Spo2',
#        'Temperature (celsius)', 'Body Battery (%)', 'Motion Intensity',
#        'Step Count', 'Total Count', 'Zero Crossing Count', 'Seconds_BBI',
#        'Seconds_stamp', 'High_ratio', 'artifacts', 'SDNN', 'RMSSD', 'SDRMSSD',
#        'TINN', 'VLF', 'LF', 'VHF', 'LFHF', 'LFn', 'PIP', 'PAS', 'GI', 'PI',
#        'C1d', 'C2d', 'DFA_alpha1', 'MFDFA_alpha1_Width', 'MFDFA_alpha1_Peak',
#        'MFDFA_alpha1_Mean', 'MFDFA_alpha1_Max', 'MFDFA_alpha1_Asymmetry',
#        'MFDFA_alpha1_Fluctuation', 'DFA_alpha2', 'MFDFA_alpha2_Width',
#        'MFDFA_alpha2_Peak', 'MFDFA_alpha2_Mean', 'MFDFA_alpha2_Max',
#        'MFDFA_alpha2_Delta', 'MFDFA_alpha2_Asymmetry',
#        'MFDFA_alpha2_Fluctuation', 'ApEn', 'MSEn', 'CMSEn', 'RCMSEn', 'KFD']

# old -- based on enhanced BBI
# based on pearson correlation
# SELECTED_FEATURES = ['Respiration (breaths/min)', 'Heart Rate (bpm)', 'Spo2',
#        'Temperature (celsius)', 'Body Battery (%)', 'Motion Intensity',
#        'Step Count', 'Total Count', 'Zero Crossing Count', 'Total Energy',
#        'Seconds_BBI', 'High_ratio', 'SDNN', 'RMSSD', 'SDRMSSD', 'MinNN',
#        'MaxNN', 'HTI', 'TINN', 'VLF', 'LF', 'VHF', 'LFHF', 'LFn', 'HFn', 'PIP',
#        'PAS', 'GI', 'PI', 'C2d', 'DFA_alpha1', 'MFDFA_alpha1_Width',
#        'MFDFA_alpha1_Peak', 'MFDFA_alpha1_Max', 'MFDFA_alpha1_Asymmetry',
#        'MFDFA_alpha1_Fluctuation', 'DFA_alpha2', 'MFDFA_alpha2_Width',
#        'MFDFA_alpha2_Peak', 'MFDFA_alpha2_Mean', 'MFDFA_alpha2_Max',
#        'MFDFA_alpha2_Delta', 'MFDFA_alpha2_Asymmetry',
#        'MFDFA_alpha2_Fluctuation', 'ApEn', 'SampEn', 'MSEn', 'CMSEn', 'RCMSEn',
#        'CD', 'KFD', 'LZC']

# based on spearman correlation
# SELECTED_FEATURES = ['Respiration (breaths/min)', 'Heart Rate (bpm)', 'Spo2',
#        'Temperature (celsius)', 'Body Battery (%)', 'Motion Intensity',
#        'Step Count', 'Total Count', 'Zero Crossing Count', 'Seconds_BBI',
#        'Seconds_stamp', 'High_ratio', 'RMSSD', 'MinNN', 'MaxNN', 'TINN', 'VHF',
#        'LFHF', 'LFn', 'PIP', 'PAS', 'GI', 'PI', 'C2d', 'DFA_alpha1',
#        'MFDFA_alpha1_Width', 'MFDFA_alpha1_Peak', 'MFDFA_alpha1_Mean',
#        'MFDFA_alpha1_Max', 'MFDFA_alpha1_Asymmetry',
#        'MFDFA_alpha1_Fluctuation', 'DFA_alpha2', 'MFDFA_alpha2_Width',
#        'MFDFA_alpha2_Peak', 'MFDFA_alpha2_Mean', 'MFDFA_alpha2_Max',
#        'MFDFA_alpha2_Delta', 'MFDFA_alpha2_Asymmetry',
#        'MFDFA_alpha2_Fluctuation', 'ApEn', 'SampEn', 'MSEn', 'CMSEn', 'RCMSEn',
#        'CD', 'KFD', 'LZC']