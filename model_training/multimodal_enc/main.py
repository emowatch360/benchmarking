# main.py
import os
import sys
import yaml
import numpy as np # type: ignore
import torch
from torch.utils.data import DataLoader, Subset # type: ignore
import pandas as pd # type: ignore
import time

from ..configs.shared_config import (data_folder, config_folder, seed) # type: ignore
from .models.conv1d import Conv1D_Encoder
from .models.resnet1d import ResNet1D_Encoder
from .models.mha1d import MultiHeadAttention1DEncoder
from .train import DGMultimodalTrainer
from .train_siamese import SiameseTrainer
from .pretraining.pretrain import SSLPretrainer
from ..utils.data_loader import CommonDataLoader, build_round_robin_indices_per_user # type: ignore
from ..utils.cross_validator import CrossValidator # type: ignore
from ..utils.common_utils import load_config, create_data_and_save_paths
from ..utils.common_metrics import compute_metrics, compute_mean_std, compute_stats_2d, perform_zscore_normalization, perform_zscore_normalization_pretraining
from ..utils.reproduce import set_seed, get_dataloader_seed_components

from .main_pretrain import perform_ssl_pretraining

cfg = load_config(os.path.join(config_folder, 'multimodal_enc.yaml'))

def perform_multimodal_enc_training(chosen_label, session, epoch=15,
                                   model_name='multimodal_enc', config=None, dataset_country='both', save=False):

    # DataLoader seed setup
    set_seed(seed) # Necessary
    seed_worker, generator = get_dataloader_seed_components(seed)

    if config:
        cfg = config
    else:
        cfg = load_config(os.path.join(config_folder, f'{model_name}.yaml'))
    dataset_name = cfg['data']['name']
    
    start_time = time.time()
    
    # File Paths - Pre-training 
    feature_description = 'features28_epoch' + str(epoch)
    numpy_pretrain_folder = os.path.join(data_folder, dataset_name, 'numpy_arrays_pretraining', '2class', f'{dataset_country}_' + feature_description)
    pretrain_cp_data_path = os.path.join(numpy_pretrain_folder, f'{dataset_country}_' + 'emowatch_pretraining.npz')
    pretrain_sp_data_path = os.path.join(numpy_pretrain_folder, f'{dataset_country}_' + 'emowatch_pretraining_sleep.npz')
    norm_params_path = os.path.join(numpy_pretrain_folder, 'zscore_params.npz')

    # DG Multimodal Model training
    # Configs
    cv_type = cfg['data']['cv']
    num_class = cfg['data']['num_classes']
    algorithm = cfg['training']['algorithm']
    require_ln = cfg['data']['require_label_normalization']
    label_type = cfg['data']['label_type']
    dg_model = cfg['model']['model_type']
    excluded_days = cfg['data']['excluded_days']
    result_root_folder = cfg['results']['root_dir']
    ssl_model = cfg['pretraining']['pretraining_type']
    ssl_encoders_freeze_epoch = cfg['training']['freeze_then_finetune_epochs']
    # File Paths
    feature_description = 'features28_epoch' + str(epoch)
    numpy_finetune_folder = os.path.join(data_folder, dataset_name, 'numpy_arrays_finetuning', '2class', f'{dataset_country}_' + feature_description)
    curr_time = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    feature_type_list = cfg['data']['feature_type_list']
    feature_tag = 'all' if feature_type_list == ['all'] else '-'.join(sorted(feature_type_list))
    results_folder = os.path.join(result_root_folder, f'{model_name}_{feature_description}_{cv_type}_{algorithm}_SSL-{ssl_model}_feat_{feature_tag}')
    data_path, all_result_path = create_data_and_save_paths(model_name=model_name,
                                                            config=cfg,
                                                            data_folder=numpy_finetune_folder,
                                                            results_folder=results_folder,
                                                            label=chosen_label,
                                                            algorithm=algorithm,
                                                            dg_model=dg_model,
                                                            change_lr=True,
                                                            curr_time=curr_time)
    
    # creating a folder to save logs
    log_folder = all_result_path[:-4]
    if os.path.exists(log_folder):
        print(f'Log folder {log_folder} already exists. Skipping this configuration.')
        return # skip if already exists
    os.makedirs(log_folder, exist_ok=True)
    # print log in a text file with datetime
    print_log_path = os.path.join(log_folder, f'training_log_{chosen_label}_{algorithm}_{curr_time}.txt')
    with open(print_log_path, 'a') as log_file:
        log_file.write('='*100+'\n')
        log_file.write(f'Data Path: {data_path}\n')
        log_file.write(f'Chosen label: {chosen_label}, session: {session}, epoch: {epoch}\n')
        log_file.write('='*100+'\n')
        yaml.dump(cfg, log_file, default_flow_style=False)
    # create teacher model template by replacing HHISS with IRM
    if algorithm.upper() == 'HHISS':
        # make sure to always run the base algorithm code before HHISS
        hhiss_base_algorithm = cfg['training']['hhiss_base_algorithm'].upper()
        teacher_model_path_template = os.path.join(log_folder, 'model_iter{}.pt')
        teacher_model_path_template = teacher_model_path_template.replace('HHISS', hhiss_base_algorithm)
    # Load train, test, and cross-validation data -- and shuffle all of them
    data_obj = CommonDataLoader(data_path, cv_type, num_class, session_filter=session,
                                label_type=label_type, # always keep it to discrete -> continuous only currently works for labels b/w 0-100
                                feature_type_list=cfg['data']['feature_type_list'],
                                require_label_normalization=require_ln,
                                zscore_path=norm_params_path,
                                require_shuffle=cfg['data']['shuffle'],
                                random_seed=seed)
    all_data_tensor, group_labels = data_obj.retrieve_2d_features(include_curr_features=cfg['data']['include_curr_features'],
                                                      include_time_index=cfg['data']['include_time_index'],
                                                      include_demo_info=cfg['data']['include_demo_info'],
                                                      include_sleep_stats=cfg['data']['include_sleep_stats'],
                                                      include_sleep_features=cfg['data']['include_sleep_features'],
                                                      include_prev_label=cfg['data']['include_prev_label'],
                                                      as_tensor=True)
    multimodal_mean, multimodal_std = data_obj.compute_mean_std_for_multimodal_data(include_time_index=cfg['data']['include_time_index'],
                                                      include_demo_info=cfg['data']['include_demo_info'],
                                                      include_sleep_stats=cfg['data']['include_sleep_stats'],
                                                      include_prev_label=cfg['data']['include_prev_label'],)

    # SSL Pre-training (at this point, encoders are loaded or pretrained from scratch)
    norm_params = (multimodal_mean[0], multimodal_std[0]) # cp data
    sleep_norm_params = (multimodal_mean[1], multimodal_std[1]) # sleep data
    pretrained_cp_encoder, pretrained_cp_aggregator = perform_ssl_pretraining(pretrain_cp_data_path, norm_params, model_name='multimodal_enc',
                                                                              data_type='cp', config=cfg, save=True)
    pretrained_sp_encoder, pretrained_sp_aggregator = perform_ssl_pretraining(pretrain_sp_data_path, sleep_norm_params, model_name='multimodal_enc',
                                                                              data_type='sp', config=cfg, save=True)
    set_seed(seed) # Also necessary for pretraining to be consistent
    
    with open(print_log_path, 'a') as log_file:
        log_file.write(f'Physiological Data size: {all_data_tensor.tensors[0].shape}\n')
        log_file.write(f'Sleep Data size: {all_data_tensor.tensors[1].shape}\n')
        log_file.write(f'Demographic (Non-temporal) Data size: {all_data_tensor.tensors[2].shape}\n')
        log_file.write(f'Label Data size: {all_data_tensor.tensors[3].shape}\n')
        log_file.write(f'Groups: {np.unique(group_labels)}\n')
    # Cross Validation
    if cv_type == 'loso':
        cv_obj = CrossValidator(cv_type=cv_type,
                                valid_size=cfg['data'][cv_type]['valid_size'],
                                print_log_path=print_log_path)
    elif cv_type == 'walk-forward':
        cv_obj = CrossValidator(cv_type=cv_type,
                                valid_size=cfg['data'][cv_type]['valid_size'],
                                start_group_idx=cfg['data'][cv_type]['start_day'], # only for time-series and walk-forward
                                temporal_window=cfg['data'][cv_type]['temporal_window'], # only for walk-forward
                                print_log_path=print_log_path)
    else:
        raise ValueError(f'CV type {cv_type} not supported for this training script!')
    results = []
    for iter, (train_idx, val_idx, test_idx) in enumerate(cv_obj.split(all_data_tensor, None, group_labels)):
        print(f'Starting Iteration {iter}!')
        iter_start_time = time.time()
        with open(print_log_path, 'a') as log_file:
            log_file.write('=' * 100+'\n');log_file.write(f'Iteration: {iter+1}\n'); log_file.write('=' * 100+'\n')
        # filter excluded days (used for zscore normalization) from test dataset
        original_test_size = len(test_idx)
        test_idx = data_obj.filter_test_indices_for_excluded_days(test_idx, excluded_days=excluded_days)
        with open(print_log_path, 'a') as log_file:
            log_file.write(f'Excluded {original_test_size - len(test_idx)} test samples from days {excluded_days}.\n')
        if len(test_idx) == 0:
            with open(print_log_path, 'a') as log_file:
                log_file.write('Skipping iteration because no test samples remain after excluding LOSO test days 1 and 2.\n')
            continue
        """ Get specific train, valid, test split """
        train_set, val_set, test_set = data_obj.create_tensor_datasets_from_indices(all_data_tensor, group_labels,
                                                                    train_idx, val_idx, test_idx, is_multimodal=True,
                                                                    print_log_path=print_log_path)
        if cfg['data']['zscore_normalization']:
            train_set = perform_zscore_normalization(train_set, mean=multimodal_mean, std=multimodal_std, is_multimodal=True)
            val_set = perform_zscore_normalization(val_set, mean=multimodal_mean, std=multimodal_std, is_multimodal=True)
            test_set = perform_zscore_normalization(test_set, mean=multimodal_mean, std=multimodal_std, is_multimodal=True)
        cv_obj.perform_assertions_on_group_labels(group_labels, train_idx, val_idx, test_idx)
        if cv_type in ['groupkfold', 'loso']: # group_labels for walk-forward would be day (so the following won't make sense)
            # only applies to training batches
            ordered_indices = build_round_robin_indices_per_user(group_labels[train_idx],
                                                                k_per_user=cfg['training']['k_per_user'],
                                                                batch_size=cfg['training']['batch_size'])
            train_set = Subset(train_set, ordered_indices)
        train_loader = DataLoader(train_set, batch_size=cfg['training']['batch_size'],
                                shuffle=False, drop_last=True) #worker_init_fn=seed_worker, generator=generator
        val_loader = DataLoader(val_set, batch_size=cfg['training']['batch_size'], shuffle=False)
        test_loader = DataLoader(test_set, batch_size=cfg['training']['batch_size'], shuffle=False)
        cfg['model']['physio_input_dim'] = list(all_data_tensor.tensors[0].shape)[-1]
        cfg['model']['sleep_input_dim'] = list(all_data_tensor.tensors[1].shape)[-1]
        cfg['model']['demo_input_dim'] = list(all_data_tensor.tensors[2].shape)[-1]
        if algorithm.upper() == 'HHISS':
            multimodal_encoder_net_obj = DGMultimodalTrainer(cfg,
                                                             pretrained_cp_encoder=pretrained_cp_encoder,
                                                             pretrained_sp_encoder=pretrained_sp_encoder,
                                                             pretrained_cp_aggregator=pretrained_cp_aggregator,
                                                             pretrained_sp_aggregator=pretrained_sp_aggregator,
                                                             teacher_model_path=teacher_model_path_template.format(iter+1),
                                                             print_log_path=print_log_path)
        elif algorithm.upper() == 'SIAMESE':
            multimodal_encoder_net_obj = SiameseTrainer(cfg, print_log_path=print_log_path)
        else:
            multimodal_encoder_net_obj = DGMultimodalTrainer(cfg, pretrained_cp_encoder=pretrained_cp_encoder,
                                                             pretrained_sp_encoder=pretrained_sp_encoder,
                                                             pretrained_cp_aggregator=pretrained_cp_aggregator,
                                                             pretrained_sp_aggregator=pretrained_sp_aggregator,
                                                             print_log_path=print_log_path)
        train_valid_metrics = multimodal_encoder_net_obj.train(train_loader=train_loader, val_loader=val_loader)
        if save:
            eval_df = pd.DataFrame(train_valid_metrics)
            num_cols = eval_df.select_dtypes(include=['number']).columns
            eval_df[num_cols] = eval_df[num_cols].round(3)
            eval_df.to_csv(os.path.join(log_folder, f'eval_results_iter{iter+1}.csv'), index=False)
            # save the trained model
            if cfg['training']['algorithm'].upper() == cfg['training']['hhiss_base_algorithm'].upper():
                model_save_path = os.path.join(log_folder, f'model_iter{iter+1}.pt')
                multimodal_encoder_net_obj.save_model(model_save_path)
        with open(print_log_path, 'a') as log_file:
            log_file.write("\n Final Evaluation on Test Set:\n")
        y_train_pred_labels, y_train_prob, y_train = multimodal_encoder_net_obj.predict_labels(train_loader)
        y_val_pred_labels, y_val_prob, y_val = multimodal_encoder_net_obj.predict_labels(val_loader)
        _, y_test_prob, y_test = multimodal_encoder_net_obj.predict_labels(test_loader)
        train_dict = compute_metrics(y_train_pred_labels, y_train_prob, y_train, 'train')
        val_dict = compute_metrics(y_val_pred_labels, y_val_prob, y_val, 'val', print_log_path=print_log_path)
        default_test_dict = compute_metrics((y_test_prob[:,1]>=0.5).astype(int), y_test_prob, y_test, 'test', prefix='default-', print_log_path=print_log_path)
        # compute threshold based on validation set
        thresh = val_dict.get('best-threshold', 0.5)
        # log the chosen threshold
        with open(print_log_path, 'a') as log_file:
            log_file.write(f'Chosen threshold based on validation set: {thresh}\n')
        test_dict = compute_metrics((y_test_prob[:,1] >= thresh).astype(int), y_test_prob, y_test, 'test', print_log_path=print_log_path)
        model_param_dict = multimodal_encoder_net_obj.return_model_param_dict()
        overall_dict = {**train_dict, **val_dict, **default_test_dict, **test_dict, **model_param_dict}
        result_row = pd.DataFrame.from_dict([overall_dict])
        results.append(result_row)
        iter_end_time = time.time()
        print(f'Iteration {iter} took {iter_end_time - iter_start_time} seconds.\n')
    all_result_df = pd.concat(results, ignore_index=True)
    mean_result_df = all_result_df.mean(axis=0)
    end_time = time.time()
    
    if save and cfg['training']['use_cp_ssl_pretraining']:
        multimodal_encoder_net_obj.save_finetuned_encoders(
            pretraining_folder=config_folder + '/../multimodal_enc/pretraining/',
            data_type='cp',
            pretraining_type=cfg['pretraining']['pretraining_type'],
            conv_layers=cfg['model']['conv_layers'],
            batch_size=cfg['pretraining']['batch_size'],
            pretrain_epochs=cfg['pretraining']['epochs'],
            feature_tag=feature_tag)
        
    if save and cfg['training']['use_sp_ssl_pretraining']:
        multimodal_encoder_net_obj.save_finetuned_encoders(
            pretraining_folder=config_folder + '/../multimodal_enc/pretraining/',
            data_type='sp',
            pretraining_type=cfg['pretraining']['pretraining_type'],
            conv_layers=cfg['model']['sp_conv_layers'],
            batch_size=cfg['pretraining']['batch_size'],
            pretrain_epochs=cfg['pretraining']['epochs'],
            feature_tag=feature_tag)
    
    with open(print_log_path, 'a') as log_file:
        log_file.write(f'Macro Avg F1-score (best threshold): {mean_result_df['macro-avg F1-score']}\n')
        log_file.write(f'AUC ROC: {mean_result_df['test-roc-macro']}\n')
        log_file.write(f'This setting took {end_time - start_time} seconds.\n')
    if save:
        all_result_df.round(3).to_csv(all_result_path, index=False)
    with open(print_log_path, 'a') as log_file:
        log_file.write('='*100)
    print('\n\nCompleted Model Training and Evaluation!\n\n')