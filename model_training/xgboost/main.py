import os
import sys
import yaml
import pickle
import numpy as np
import pandas as pd
from collections import Counter
# Custom Methods
from xgboost_utils import (
        compute_accuracy_dict,
        compute_roc_dict,
        compute_training_loss_dict,
        compute_precision_recall_dict
)
from train import XGBoostTrainer
# Config
sys.path.append('../configs/')
sys.path.append('../utils/')
from data_loader import CommonDataLoader
from cross_validator import CrossValidator
from shared_config import (data_folder, config_folder, seed) # type: ignore
from common_utils import load_config, create_data_and_save_paths # type: ignore
from reproduce import set_seed

set_seed(seed)

def perform_xgboost_training(chosen_label, session, epoch,
                             cluster_feature='', cluster=-1, 
                             model_name='xgboost', config=None, dataset_country='both', save=False,
                             numpy_folder=None,
                             results_folder=None):
    if config:
        cfg = config
    else:
        cfg = load_config(os.path.join(config_folder, f'{model_name}.yaml'))
    # Dataset Path
    dataset_name = cfg['data']['name']
    cv_type = cfg['data']['cv']
    num_class = cfg['data']['num_classes']
    excluded_days = cfg['data']['excluded_days']
    result_root_folder = cfg['results']['root_dir']
    # modified
    feature_description = f'features28_epoch{epoch}'
    if numpy_folder is None:
        numpy_folder = f'/data/emowatch/emowatch/numpy_arrays_finetuning/2class/{dataset_country}_features28_epoch{epoch}/'
    if results_folder is None:
        results_folder = os.path.join(result_root_folder, f'{dataset_country}', f'{model_name}_{feature_description}_{cv_type}')
    data_path, all_result_path = create_data_and_save_paths(model_name=model_name,
                                                            config=cfg,
                                                            data_folder=numpy_folder,
                                                            results_folder=results_folder,
                                                            label=chosen_label)
    if cluster != -1:
        data_path = data_path[:-33] + f'{cluster_feature}_cluster{cluster}_' + data_path[-33:]
        all_result_path = all_result_path[:-7] + f'{cluster_feature}_cluster{cluster}_all.csv'
    if session != 'all':
        all_result_path = all_result_path[:-7] + f'{session}.csv'
    os.makedirs(os.path.dirname(all_result_path), exist_ok=True)
    log_folder = all_result_path[:-4]
    os.makedirs(log_folder, exist_ok=True)
    print_log_path = os.path.join(log_folder, f'training_log_{chosen_label}_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.txt')
    with open(print_log_path, 'a') as log_file:
        log_file.write('='*100+'\n')
        log_file.write(f'Chosen label: {chosen_label}, session: {session}, epoch: {epoch}\n')
        # write data_path
        log_file.write(f'Data path: {data_path}\n')
        log_file.write('='*100+'\n')
        yaml.dump(cfg, log_file, default_flow_style=False)
    # File Paths
    params = cfg['training']
    # Load train, test, and cross-validation data -- and shuffle all of them
    data_obj = CommonDataLoader(data_path, cv_type, num_class,
                                session_filter=session,
                                require_shuffle=cfg['data']['shuffle'],
                                random_seed=seed)
    X_data, group_labels = data_obj.retrieve_1d_features(
                                                    include_curr_features=cfg['data']['include_curr_features'],
                                                    include_time_index=cfg['data']['include_time_index'],
                                                    include_demo_info=cfg['data']['include_demo_info'],
                                                    include_sleep_stats=cfg['data']['include_sleep_stats'],
                                                    include_sleep_features=cfg['data']['include_sleep_features'],
                                                    include_prev_label=cfg['data']['include_prev_label']
                                                    )
    with open(print_log_path, 'a') as log_file:
        log_file.write(f'Data size: {X_data.shape}\n')
        log_file.write(f'Groups: {np.unique(group_labels)}\n')
        log_file.write(f'{params}\n')
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
        raise ValueError(f'Unsupported CV type: {cv_type}')

    results = []
    for iter, (train_idx, valid_idx, test_idx) in enumerate(cv_obj.split(X_data, None, group_labels)):
        with open(print_log_path, 'a') as log_file:
            log_file.write('=' * 100+'\n');log_file.write(f'Iteration: {iter+1}\n'); log_file.write('=' * 100+'\n')
        if cv_type in ['loso', 'lodo']:
            assert (train_idx.size + valid_idx.size + test_idx.size) == group_labels.size
        # filter excluded days from test dataset
        original_test_size = len(test_idx)
        test_idx = data_obj.filter_test_indices_for_excluded_days(test_idx, excluded_days=excluded_days)
        with open(print_log_path, 'a') as log_file:
            log_file.write(f'Excluded {original_test_size - len(test_idx)} test samples from days {excluded_days}.\n')
        if len(test_idx) == 0:
            with open(print_log_path, 'a') as log_file:
                log_file.write('Skipping iteration because no test samples remain after excluding configured test days.\n')
            continue
        """ Get specific train, valid, test split """
        X_train, X_valid, X_test = X_data[train_idx], X_data[valid_idx], X_data[test_idx]
        y_train, y_valid, y_test = data_obj.compute_labels(train_idx, valid_idx, test_idx)
        cv_obj.perform_assertions_on_group_labels(group_labels, train_idx, valid_idx, test_idx)
        """ Train model based on loaded data and params """
        model_obj = XGBoostTrainer(params, num_class)
        model_obj.train(X_train, y_train, X_valid, y_valid)
        # save model
        # if save:
        #     pickle.dump(model_obj.model, open(os.path.join(log_folder, f'xgboost_model_{chosen_label}.pkl'), 'wb'))
        #     # save the test data as sample test data
        #     pickle.dump({'X_test': X_test, 'y_test': y_test}, open(os.path.join(log_folder, f'sample_test_data_{chosen_label}.pkl'), 'wb'))
        #     # pickle.dump(model_obj.model, open(os.path.join(log_folder, f'xgboost_model_{chosen_label}_iter{iter+1}.pkl'), 'wb'))
        eval_results = model_obj.get_eval_results()
        # save eval_results into csv using pandas
        # eval results is a dict with two keys validation_0 and validation_1, each with two keys, 'mlogloss' and 'auc'
        if save:
            eval_df = pd.DataFrame.from_dict(eval_results)
            eval_df.reset_index(inplace=True)
            eval_df.rename(columns={'index': 'metric'}, inplace=True)
            eval_df.to_csv(os.path.join(log_folder, f'eval_results_iter{iter+1}.csv'), index=False)
        best_iteration = model_obj.get_best_iteration()
        with open(print_log_path, 'a') as log_file:
            log_file.write(f'Best Iteration: {best_iteration}\n')
            log_file.write(f'Initial Log Loss: {eval_results["validation_0"]["mlogloss"][0]} | {eval_results["validation_1"]["mlogloss"][0]}\n')
            log_file.write(f'Final Log Loss: {eval_results["validation_0"]["mlogloss"][best_iteration]} | {eval_results["validation_1"]["mlogloss"][best_iteration]}\n')
        fi_df, fi_top10_dict = model_obj.compute_feature_importance()
        with open(print_log_path, 'a') as log_file:
            log_file.write('Feature Importance:\n')
            # indexing starts from f0
            log_file.write(f'{fi_df[:20].to_string(index=False)}\n')
        if save:
            # saved in a sorted manner
            fi_df.to_csv(os.path.join(log_folder, f'fi_dict_iter{iter+1}.csv'), index=False)
        """ Compute train, valid, test predictions """
        y_train_pred_score = model_obj.evaluate(X_train)
        y_valid_pred_score = model_obj.evaluate(X_valid)
        y_test_pred_score = model_obj.evaluate(X_test)
        if num_class == 2:
            chosen_threshold = 0.5 #model_obj.determine_2class_threshold(y_valid, y_valid_pred_score) - setting it to 0.5 given the small size of validation dataset
            with open(print_log_path, 'a') as log_file:
                log_file.write(f'Chosen Threshold based on validation set: {chosen_threshold}\n')
        else:
            chosen_threshold = -1
        y_train_pred_labels = model_obj.convert_scores_to_labels(y_train_pred_score)
        y_valid_pred_labels = model_obj.convert_scores_to_labels(y_valid_pred_score)
        y_test_pred_labels = model_obj.convert_scores_to_labels(y_test_pred_score)
        # save pid labels, y_test, y_test_pred_labels using pandas into a csv file
        if save:
            pred_df = pd.DataFrame({
                'pid': data_obj.get_pid_labels(test_idx),
                'y_test': y_test,
                'y_test_pred_labels': y_test_pred_labels,
                'label0_pred_prob': y_test_pred_score[:, 0],
                'label1_pred_prob': y_test_pred_score[:, 1],
                'time_index': data_obj.get_time_indices(test_idx)
            })
            pred_df.to_csv(os.path.join(log_folder, f'predictions_iter{iter+1}.csv'), index=False)
        # compute accuracy
        train_accuracy, train_roc, _ = model_obj.compute_metrics(y_train, y_train_pred_labels,
                                                            y_train_pred_score, chosen_label, 'train')
        valid_accuracy, valid_roc, _ = model_obj.compute_metrics(y_valid, y_valid_pred_labels,
                                                            y_valid_pred_score, chosen_label, 'valid')
        test_accuracy, test_roc, test_metrics = model_obj.compute_metrics(y_test, y_test_pred_labels,
                                                            y_test_pred_score, chosen_label, 'test', print_log_path=print_log_path)
        with open(print_log_path, 'a') as log_file:
            log_file.write(f'Train: ROC-AUC:{train_roc}| Valid: ROC-AUC:{valid_roc} | Test: ROC-AUC:{test_roc}\n')
        # create accuracy, roc, training-loss dicts
        acc_dict = compute_accuracy_dict(train_accuracy, valid_accuracy, test_accuracy)
        roc_dict = compute_roc_dict(train_roc, valid_roc, test_roc, num_class)
        training_loss_dict = compute_training_loss_dict(eval_results, ['validation_0', 'validation_1'],
                                                        'mlogloss', best_iteration, chosen_threshold)
        # create individual precision/recall/support dict
        precision_recall_dict = compute_precision_recall_dict(test_metrics, num_class, chosen_label)
        metrics_dict = {**roc_dict, **precision_recall_dict, **training_loss_dict, **fi_top10_dict}
        result_row = pd.DataFrame.from_dict(metrics_dict)
        results.append(result_row)
    all_result_df = pd.concat(results, ignore_index=True)
    mean_result = all_result_df['macro-avg F1-score'].mean(axis=0)
    with open(print_log_path, 'a') as log_file:
        log_file.write(f'Macro Avg F1-score across all folds: {mean_result}\n')
    if save:
        num_cols = all_result_df.select_dtypes(include=['number']).columns
        all_result_df[num_cols] = all_result_df[num_cols].round(3)
        all_result_df.to_csv(all_result_path, index=False)