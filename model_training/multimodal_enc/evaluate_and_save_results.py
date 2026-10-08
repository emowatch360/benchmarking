import os
from ..configs.shared_config import config_folder
from ..utils.common_utils import load_config
from .main import perform_multimodal_enc_training

# Example file showcasing how to run multiple training rounds

labels_of_interest = ['arousal_current'] #, 'valence_current']
feature_dim = '2D' # ,'1D-stat']
inf_script = 'main.py'
epoch = '30'
cv_type = 'walk-forward' # 'walk-forward' or 'loso'
setting = 'single_run' # 'feature_combination_eval' or 'hyperparameter_tuning' or 'single_run'
dataset_country = 'sg' # options: both, jp, sg

algorithms_list = ['ERM'] #['ERM','VREX','IRM','DRO','Siamese','DANN','HHISS']
config = load_config(os.path.join(config_folder, 'multimodal_enc.yaml'))
config['data']['feature_dim'] = feature_dim
assert config['data']['cv'] == cv_type, f"CV type in config ({config['data']['cv']}) does not match the specified cv_type ({cv_type})."

if setting == 'hyperparameter_tuning':
	session = 'all'
	# for walk-forward, don't add layers over 64 units
	if cv_type == 'walk-forward':
		conv_layers_dict = {'cnn' : [[32], [16], [8], [32, 64], [16, 32], [8, 16]], 
							'resnet': [[32], [16], [8]],
							'mha': [[32], [16], [8]]}
	elif cv_type == 'loso':
		conv_layers_dict = {'cnn' : [[128], [64], [32], [32, 64], [64, 128], [32, 64, 128]], #, [16, 32, 64], [8, 16, 32]],
						'resnet': [[128], [64], [32]],
						'mha': [[128], [64], [32]]} 
	
	feature_dict_list = [
	{'include_curr_features': True, 'include_sleep_stats': False, 'include_sleep_features': False, 'include_time_index': False, 'include_demo_info': False, 'sleep_normalization': False},
	]
		
	for chosen_label in labels_of_interest:
		for algorithm in algorithms_list:
			config['training']['algorithm'] = algorithm
			for model_type, conv_layers_list in conv_layers_dict.items():
				config['model']['model_type'] = model_type
				for conv_layers in conv_layers_list:
					config['model']['conv_layers'] = conv_layers # set sp_conv_layers as well to change layers for sleep data
					print(conv_layers)
					for feature_dict in feature_dict_list:
						print('='*100)
						print(f'Label: {chosen_label}, Algorithm: {algorithm}, Feature Dim: {feature_dim}, Session: {session}, Model: {model_type}, Layers: {conv_layers}, Features: {feature_dict}')
						for key, val in feature_dict.items():
							config['data'][key] = val
						if session == 'all':
							perform_multimodal_enc_training(chosen_label, session, epoch, model_name='multimodal_enc', config=config,
									 					dataset_country=dataset_country, save=True)

if setting == 'single_run':
	session = 'all'
	# for walk-forward, don't add layers over 64 units
	if cv_type == 'walk-forward':
		conv_layers_dict = {#'cnn' : [16], 
							#'resnet': [8],
							'mha': [16]}
	elif cv_type == 'loso':
		conv_layers_dict = {'cnn' : [16, 32], 
						#'resnet': [16],
						#'mha': [32]
						} 
	
	feature_dict_list = [
	{'include_curr_features': True, 'include_sleep_stats': True, 'include_sleep_features': True, 'include_time_index': True, 'include_demo_info': True, 'sleep_normalization': False,
  		'zscore_normalization': True},
	]
		
	for chosen_label in labels_of_interest:
		for algorithm in algorithms_list:
			config['training']['algorithm'] = algorithm
			for model_type in conv_layers_dict.keys():
				config['model']['model_type'] = model_type
				config['model']['conv_layers'] = conv_layers_dict[model_type]
				config['model']['sp_conv_layers'] = conv_layers_dict[model_type] # for single run, set sp_conv_layers same as conv_layers
				print(f'Model type: {model_type}, Conv layers: {conv_layers_dict[model_type]}, SP Conv layers: {conv_layers_dict[model_type]}')
				for feature_dict in feature_dict_list:
					print('='*100)
					print(f'Label: {chosen_label}, Algorithm: {algorithm}, Feature Dim: {feature_dim}, Session: {session}, Features: {feature_dict}')
					for key, val in feature_dict.items():
						config['data'][key] = val
					if session == 'all':
						perform_multimodal_enc_training(chosen_label, session, epoch, model_name='multimodal_enc', config=config,
									 					dataset_country=dataset_country, save=True)

elif setting == 'feature_combination_eval':
	if cv_type == 'walk-forward':
		conv_layers_dict = {'cnn' : [16], 
							'resnet': [8],
							'mha': [16]}
	elif cv_type == 'loso':
		conv_layers_dict = {'cnn' : [16, 32], 
						'resnet': [16],
						'mha': [32]} 
	algorithm = 'ERM'
	session = 'all'
	config['training']['algorithm'] = algorithm
	feature_list = ['include_curr_features', 'include_sleep_stats', 'include_sleep_features', 'include_time_index', 'include_demo_info']
	n_options = 2**len(feature_list)
	feature_dict_list = []
	for i in range(1, n_options): # ignoring the first option where no features are included
		num = i
		curr_dict = {}
		for j, feature in enumerate(feature_list):
			if num % 2 == 1:
				curr_dict[feature] = True
			else:
				curr_dict[feature] = False
			num = num // 2
		assert num == 0, "Error in generating feature inclusion combinations."
		curr_dict['zscore_normalization'] = True # ensuring only one of these is True
		feature_dict_list.append(curr_dict)
	for chosen_label in labels_of_interest:
		# for model_type, conv_layers_list in conv_layers_dict[chosen_label].items():
		for model_type in conv_layers_dict.keys():
			config['model']['model_type'] = model_type
			config['model']['conv_layers'] = conv_layers_dict[model_type]
			config['model']['sp_conv_layers'] = conv_layers_dict[model_type]
			print(f'Model type: {model_type}, Conv layers: {conv_layers_dict[model_type]}, SP Conv layers: {conv_layers_dict[model_type]}')
			for feature_dict in feature_dict_list:
				print('='*100)
				print(f'Label: {chosen_label}, Features: {feature_dict}')
				for key, val in feature_dict.items():
					config['data'][key] = val
				perform_multimodal_enc_training(chosen_label, session, epoch, model_name='multimodal_enc', config=config,
									 					dataset_country=dataset_country, save=True)