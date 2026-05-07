import os
import sys
import random
sys.path.append('../configs/')
sys.path.append('../utils/')
from shared_config import config_folder #type: ignore
from common_utils import load_config #type: ignore
from main import perform_xgboost_training
from predict_per_session import compute_per_session_performance
# set seed for random
random.seed(42)

param_space = {
    "max_depth":        [3, 4, 5],
    "min_child_weight": [1, 5, 10],
    "gamma":            [0, 1],
    "subsample":        [0.7, 0.8, 0.9],
    "colsample_bytree": [0.6, 0.7, 0.8],
    "reg_alpha":        [0.0, 0.5, 1.0],
    "reg_lambda":       [1.0, 3.0, 6.0],
}

cv_type = 'walk-forward' # 'walk-forward' or 'loso'
total_iterations = 20
labels_of_interest = ['arousal_current', 'valence_current'] #, 'stressed_current'] #['arousal_current', 'valence_current'] #, 'stressed_current'] #,  'valence_current'] #['stressed_current'] #
session = 'all' #['morning', 'noon', 'evening', 'bedtime'] # # #['all'] #
feature_dim = '1D-stat'
epoch ='30' #['30', '15', '10']
config = load_config(os.path.join(config_folder, 'xgboost.yaml'))
assert config['data']['cv'] == cv_type, f"CV type in config ({config['data']['cv']}) does not match the specified cv_type ({cv_type})."
config['data']['feature_dim'] = feature_dim
setting = 'feature_combination_eval' # 'final_combination_eval' or 'hyperparameter_tuning'

if setting == 'feature_combination_eval':
	if cv_type == 'walk-forward':
		param_dict = {'valence_current': {'max_depth':5,'min_child_weight':1,'gamma':0,'subsample':0.8,'colsample_bytree':0.7,'reg_alpha':0.5,'reg_lambda':6.0},
					'arousal_current': {'max_depth':5,'min_child_weight':5,'gamma':0,'subsample':0.7,'colsample_bytree':0.6,'reg_alpha':0.0,'reg_lambda':1.0}}
					# 'stressed_current': {'max_depth':5,'min_child_weight':5,'gamma':0,'subsample':0.9,'colsample_bytree':0.7,'reg_alpha':1.0,'reg_lambda':1.0}}
	elif cv_type == 'loso':
		# to be updated
		param_dict = {'valence_current': {'max_depth':5,'min_child_weight':5,'gamma':0,'subsample':0.9,'colsample_bytree':0.7,'reg_alpha':1.0,'reg_lambda':1.0},
					'arousal_current': {'max_depth':5,'min_child_weight':1,'gamma':0,'subsample':0.7,'colsample_bytree':0.8,'reg_alpha':1.0,'reg_lambda':1.0}}
					# 'stressed_current': {'max_depth':4,'min_child_weight':1,'gamma':1,'subsample':0.9,'colsample_bytree':0.8,'reg_alpha':0.5,'reg_lambda':6.0}}
	
	# create a power set of feature inclusion options
	feature_list = ['include_curr_features', 'include_time_index', 'include_demo_info', 'include_sleep_stats', 'include_sleep_features']
	n_options = 2**len(feature_list)
	feature_dict_list = []
	for i in range(1, n_options): # ignoring the first option where no features are included
		num = i
		curr_dict = {'sleep_normalization': False} # default values for features not in feature_list
		for j, feature in enumerate(feature_list):
			if num % 2 == 1:
				curr_dict[feature] = True
			else:
				curr_dict[feature] = False
			num = num // 2
		assert num == 0, "Error in generating feature inclusion combinations."
		feature_dict_list.append(curr_dict)
	for chosen_label in labels_of_interest:
		for key, val in param_dict[chosen_label].items():
			config['training'][key]=val
		for feature_dict in feature_dict_list:
			print('='*100)
			print(f'Label: {chosen_label}, Epoch: {epoch}')
			for key, val in feature_dict.items():
				config['data'][key] = val
			assert session ==  'all', "Only 'all' session is supported in this script."
			perform_xgboost_training(chosen_label, session, epoch,
										model_name='xgboost', config=config, save=True)

elif setting == 'hyperparameter_tuning':
	feature_dict_list = [{'include_curr_features': True, 'include_sleep_stats': False, 'include_sleep_features': False, 
					   'include_time_index': False, 'include_demo_info': False, 'sleep_normalization': False}]
	for iter_num in range(total_iterations):
		for k, v in param_space.items():
			config['training'][k] = random.choice(v)
		for chosen_label in labels_of_interest:
			for feature_dict in feature_dict_list:
				print('='*100)
				print(f'Label: {chosen_label}, Epoch: {epoch}')
				for key, val in feature_dict.items():
					config['data'][key] = val
				assert session ==  'all', "Only 'all' session is supported in this script."
				perform_xgboost_training(chosen_label, session, epoch,
										 model_name='xgboost', config=config, save=True)




