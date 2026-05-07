import os

def get_dataset_dict(dataset_name):
    ds_info = {}
    if dataset_name == 'emowatch':
        ds_info['root-folder'] = '/data/emowatch/'
        # ds_info['chosen-label'] = 'stressed_current'
        ds_info['dataset-country'] = 'both'
        ds_info['feature-dim'] = '1D'
        # ds_info['duration'] = 30 #120 for 2D-stride, 30 for rest
        ds_info['sleep-2d-chunks'] = 30 # keep this to 30 for now
        ds_info['time-offset'] = 3 # offset for causality (HRV feature computation peeks a little ahead for moving average, and artifact correction)
        # ds_info['2D-stride'] = {'window-length': 30,
        #                         'hop-length': 10}
        ds_info['num-classes'] = 2
        ds_info['data-folder'] = os.path.join(ds_info['root-folder'], 'emowatch')
        # ds_info['processed-feature-folder'] = os.path.join(ds_info['data-folder'], 'processed_data_all')
        ds_info['processed-feature-folder'] = os.path.join(ds_info['data-folder'], 'processed_data_all_filtered')
        ds_info['label-path'] = os.path.join(ds_info['data-folder'], 'processed_labels_all.csv')
        # ds_info['save-folder'] = os.path.join(ds_info['data-folder'], 'numpy_arrays', f'features29_epoch{ds_info['duration']}')
        # ds_info['demographics-path'] = os.path.join(ds_info['data-folder'], 'demographics-info-modified.csv')
        ds_info['demographics-path'] = os.path.join(ds_info['data-folder'], 'demographics-info-all-modified.csv')
        ds_info['c1-dates-path'] = os.path.join(ds_info['data-folder'], f'detailed_c1_start_end_dates.csv')
        ds_info['c2-dates-path'] = os.path.join(ds_info['data-folder'], f'c2_start_end_dates.csv')
        # ds_info['sleep-feat-path'] = os.path.join(ds_info['data-folder'], 'processed_sleep_data_all')
        ds_info['sleep-feat-path'] = os.path.join(ds_info['data-folder'], 'processed_sleep_data_all_filtered')
        ds_info['sleep-stat-path'] = os.path.join(ds_info['data-folder'], 'sleep_summary_all.csv')
        # ds_info['cluster-feature'] = 'RMSSD'
        # ds_info['cluster-path'] = os.path.join(ds_info['data-folder'], 'sleep_clusters', f'sleep_cluster_{ds_info['cluster-feature']}.csv')
        # ensure all day lists are in order (mainly for prev survey label computation)
        # orig_pid_list = ['MM_'+f'{pid:02d}' for pid in range(1, 46)] + ['pid'+f'{idx:02d}' for idx in range(1, 35)] + ['c2com'+f'{idx:02d}' for idx in range(1, 20)]
        if ds_info['dataset-country'] == 'c1':
            orig_pid_list = ['MM_'+f'{pid:02d}' for pid in range(1, 46)]
            ds_info['label-type'] = 'continuous'
        elif ds_info['dataset-country'] == 'c2':
            orig_pid_list = ['c2com'+f'{idx:02d}' for idx in range(1, 20)] + ['pid'+f'{idx:02d}' for idx in range(1, 35)]
            ds_info['label-type'] = 'likert' # or 'likert'
        pid_to_remove = ['MM_12', 'c2com02', 'c2com16', 'pid02', 'pid04', 'pid05', 'pid07', 'pid11', 'pid14', 'pid16', 'pid18', 'pid24', 'pid28', 'pid29', 'pid30', 'pid33', 'pid34', 'pid32']
        ds_info['pid-list'] = [pid for pid in orig_pid_list if pid not in pid_to_remove]
        ds_info['day-list'] = ['d'+f'0{idx}'[-2:] for idx in range(1, 29)]
        ds_info['total-pid-num'] = len(ds_info['pid-list'])
        ds_info['total-days'] = 28
        # ds_info['max-score'] = 100
        ds_info['features-of-interest'] = ['Respiration (breaths/min)', 'Heart Rate (bpm)', 'Spo2',
                                            'Temperature (celsius)', #'Body Battery (%)',
                                            'Motion Intensity',
                                            'Step Count', 'Total Count', 'Zero Crossing Count', 
                                                # time domain features below
                                            'SDNN', 'RMSSD', 'SDRMSSD', 'TINN',
                                            # frequency features below
                                            'VLF', 'LF', 'VHF', 'LFHF', 'LFn', 
                                            # non-linear features below
                                            'PIP', 'PAS', 'GI', 'PI',
                                            'C1d', 'C2d', 
                                            'ApEn', 'MSEn', 'CMSEn', 'RCMSEn', 'HFD']
        ds_info['count-based-features'] = ['Step Count', 'Total Count', 'Zero Crossing Count']
        # c1 data list
    #     ['Respiration (breaths/min)', 'Heart Rate (bpm)', 'Spo2',
    #    'Temperature (celsius)', 'Body Battery (%)', 'Motion Intensity',
    #    'Step Count', 'Total Count', 'Zero Crossing Count', 
    # #    'Seconds_BBI','Seconds_stamp', 'High_ratio', 'artifacts', 
    #     # time domain features below
    #    'SDNN', 'RMSSD', 'SDRMSSD', 'TINN',
    #    # frequency features below
    #    'VLF', 'LF', 'VHF', 'LFHF', 'LFn', 
    #    # non-linear features below
    #    'PIP', 'PAS', 'GI', 'PI',
    #    'C1d', 'C2d', 
    # #    'DFA_alpha1', 'MFDFA_alpha1_Width', 'MFDFA_alpha1_Peak',
    # #    'MFDFA_alpha1_Mean', 'MFDFA_alpha1_Max', 'MFDFA_alpha1_Asymmetry',
    # #    'MFDFA_alpha1_Fluctuation', 'DFA_alpha2', 'MFDFA_alpha2_Width',
    # #    'MFDFA_alpha2_Peak', 'MFDFA_alpha2_Mean', 'MFDFA_alpha2_Max',
    # #    'MFDFA_alpha2_Delta', 'MFDFA_alpha2_Asymmetry',
    # #    'MFDFA_alpha2_Fluctuation', 
    #    'ApEn', 'MSEn', 'CMSEn', 'RCMSEn', 'KFD']
        ds_info['interp'] = True # interpolation
        ds_info['interp-prop'] ={'method': 'linear',
                                 'axis': 0,
                                 'limit_direction': 'forward',
                                 'limit': 5}
        # sleep related
        # ds_info['sleep-stat-path'] = os.path.join(ds_info['data-folder'], 'sleep_summary_cluster10d.csv')
        ds_info['sleep-stat-features'] = ['duration', 'Rem_per', 'Deep_per', 'Light_per', 'Awake_per', 'start_time', 'end_time'] #, 'RMSSD']
        ds_info['sleep-features-of-interest'] = ds_info['features-of-interest'] # same as current features
                                                # ['Respiration (breaths/min)', 'Heart Rate (bpm)', 'Spo2',
                                                # 'Temperature (celsius)', #'Body Battery (%)',
                                                # 'Motion Intensity',
                                                # 'Step Count', 'Total Count', 'Zero Crossing Count', 
                                                #     # time domain features below
                                                # 'SDNN', 'RMSSD', 'SDRMSSD', 'TINN',
                                                # # frequency features below
                                                # 'VLF', 'LF', 'VHF', 'LFHF', 'LFn', 
                                                # # non-linear features below
                                                # 'PIP', 'PAS', 'GI', 'PI',
                                                # 'C1d', 'C2d', 
                                                # 'ApEn', 'MSEn', 'CMSEn', 'RCMSEn', 'HFD']
        # ds_info['sleep-cluster-features'] = ['MeanNN', 'RMSSD', 'SDSD', 'CVSD', 'MedianNN', 'MadNN', 'IQRNN',
        #                                 'Prc20NN', 'Prc80NN', 'pNN50', 'pNN20', 'MinNN', 'MaxNN', 'HTI', 'HF',
        #                                 'VHF', 'TP', 'LnHF', 'SD1', 'S', 'CVI', 'PIP', 'IALS', 'PSS', 'PAS',
        #                                 'SD1d', 'SD1a','ApEn', 'Spo2', 'Total Energy',
        #                                 'Zero Crossing Count', 'Time Above Threshold',
        #                                 'Respiration (breaths/min)']
        # ds_info['sleep-feat-path'] = os.path.join(ds_info['data-folder'], 'processed_sleep_data_c1')
        # ds_info['offset-features'] = False # apply change-score normalization
        ds_info['total-minutes'] = 1440 # 24 hours in minutes
        ds_info['night-minutes'] = 959 # 15:59 in minutes -- to include evening and bedtime
        ds_info['default-sleep-time-index'] = 22*60 # 10pm in minutes
        ds_info['default-wake-time-index'] = 10*60 # 10am in minutes
    return ds_info