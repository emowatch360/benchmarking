import os
import pandas as pd
from datetime import datetime, timedelta
from utils import compute_day_id #type:ignore

features = 'MeanNN,SDNN,SDANN1,SDNNI1,RMSSD,SDSD,CVNN,CVSD,MedianNN,MadNN,MCVNN,IQRNN,SDRMSSD,Prc20NN,Prc80NN,pNN50,pNN20,MinNN,MaxNN,HTI,TINN,VLF,LF,HF,VHF,TP,LFHF,LFn,HFn,LnHF,SD1,SD2,SD1SD2,S,CSI,CVI,CSI_Modified,PIP,IALS,PSS,PAS,GI,SI,AI,PI,C1d,C1a,SD1d,SD1a,C2d,C2a,SD2d,SD2a,Cd,Ca,SDNNd,SDNNa,DFA_alpha1,MFDFA_alpha1_Width,MFDFA_alpha1_Peak,MFDFA_alpha1_Mean,MFDFA_alpha1_Max,MFDFA_alpha1_Delta,MFDFA_alpha1_Asymmetry,MFDFA_alpha1_Fluctuation,MFDFA_alpha1_Increment,DFA_alpha2,MFDFA_alpha2_Peak,MFDFA_alpha2_Mean,MFDFA_alpha2_Max,MFDFA_alpha2_Asymmetry,MFDFA_alpha2_Fluctuation,MFDFA_alpha2_Increment,ApEn,SampEn,ShanEn,FuzzyEn,MSEn,CMSEn,RCMSEn,CD,HFD,KFD,LZC,Spo2,Total Energy,Zero Crossing Count,respiration'
feature_list = features.split(',')
first_date_str = '2024-11-04'

# Load input CSV
data_folder = '/data/emowatch/'
input_path = os.path.join(data_folder, 'clusters_different_features_rawBBI.csv')
output_path = os.path.join(data_folder, 'sleep_clusters', 'sleep_cluster_all_features.csv')
df = pd.read_csv(input_path)
# remove rows when 'end' feature is empty or 'error
df = df[(df['end'].notna()) & (df['end'] != '') & (df['end'] != 'error')]
# Compute 'Next Day' for each row
df['Next Day'] = df['end'].apply(lambda x: compute_day_id(x.split(' ')[0], first_date_str))
# Rename and select relevant columns
df_output = df.rename(columns={'user_id': 'PID'})[['PID', 'Next Day', *feature_list]]
# df_output = df_output.rename(columns={'cluster': 'Cluster'})
# Save to new CSV
df_output.to_csv(output_path, index=False)