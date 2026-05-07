import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from collections import Counter
# Custom Methods
from xgboost_utils import (
        compute_precision_recall_dict
)
from train import XGBoostTrainer
# Config
sys.path.append('../configs/')
sys.path.append('../utils/')
from shared_config import (data_folder, config_folder, seed) # type: ignore
from common_utils import load_config, create_data_and_save_paths # type: ignore
from reproduce import set_seed
set_seed(seed)

def compute_per_session_performance(chosen_label, cluster_feature,
                                    cluster, session, epoch, model_name='xgboost',
                                    config=None, save=False,
                                    num_iter=18):
    if config:
        cfg = config
    else:
        cfg = load_config(os.path.join(config_folder, f'{model_name}.yaml'))
    # Dataset Path
    num_class = cfg['data']['num_classes']
    params = cfg['training']
    cv_type = cfg['data']['cv']
    # modified
    feature_description = f'features29_epoch{epoch}_unmatched_{cv_type}'
    _, all_result_path = create_data_and_save_paths(model_name=model_name,
                                                            config=cfg,
                                                            data_folder='',
                                                            results_folder=f'../results/{model_name}_{feature_description}',
                                                            label=chosen_label)
    if cluster != -1:
        all_result_path = all_result_path[:-7] + f'{cluster_feature}_cluster{cluster}_all.csv'
    assert session != 'all', 'Expecting a specific session, not "all"'
    session_result_path = all_result_path[:-7] + f'{session}.csv'
    # read the all results, and get the session results
    all_pred_template = all_result_path.replace('.csv', '/predictions_iter{}.csv')
    model_obj = XGBoostTrainer(params, num_class)
    results = []
    for idx in range(num_iter):
        all_pred_path = all_pred_template.format(idx+1)
        all_results_idx = pd.read_csv(all_pred_path)
        # read the time_index column, create a new column called session
        # if 240 <= time_index < 660, session is 'morning' 4:00 to 11:00
        # if 660 <= time_index < 900, session is 'noon' # 11:00 to 15:00
        # if 900 <= time_index < 1140, session is 'evening' # 15:00 to 19:00
        # if 1140 <= time_index < 1440 or time_index < 240, session is 'bedtime' # 19:00 to 24:00 or 0:00 to 4:00
        all_results_idx['session'] = 'bedtime'
        all_results_idx.loc[(all_results_idx['time_index'] >= 240) & (all_results_idx['time_index'] < 660), 'session'] = 'morning'
        all_results_idx.loc[(all_results_idx['time_index'] >= 660) & (all_results_idx['time_index'] < 900), 'session'] = 'noon'
        all_results_idx.loc[(all_results_idx['time_index'] >= 900) & (all_results_idx['time_index'] < 1140), 'session'] = 'evening'
        # count the number of elements of each of the four sessions
        session_counts = Counter(all_results_idx['session'])
        print(f'Session counts for {all_pred_path}')
        print(f'{session_counts}')
        # check if the session exists in the dataframe
        # filter the dataframe to only include rows where 'session' matches the session
        session_results = all_results_idx[all_results_idx['session'] == session]
        if session_results.empty:
            print(f'No results found for session {session} in {all_pred_path}')
            continue
        # compute results
        y_test = session_results['y_test'].values.astype(int)
        y_test_pred_labels = session_results['y_test_pred_labels'].values.astype(int)
        # combine label0_pred_prob and label1_pred_prob into a single array to get y_test_pred_score
        y_test_pred_score = np.vstack((session_results['label0_pred_prob'].values,
                                       session_results['label1_pred_prob'].values)).T
        # check that every row in y_test_pred_score adds upto 1
        if not np.allclose(y_test_pred_score.sum(axis=1), 1):
            raise ValueError("y_test_pred_score rows do not sum to 1. Check the predictions.")
        assert np.all(np.argmax(y_test_pred_score, axis=1) - y_test_pred_labels == 0)
        _, test_roc, test_metrics = model_obj.compute_metrics(y_test, y_test_pred_labels,
                                                            y_test_pred_score, chosen_label, 'test')
        print(f'Test: ROC-AUC:{test_roc}')
        # create accuracy, roc, training-loss dicts
        roc_dict = {'test-roc-macro': test_roc['macro']}
        # create individual precision/recall/support dict
        precision_recall_dict = compute_precision_recall_dict(test_metrics, num_class, chosen_label)
        metrics_dict = {**roc_dict, **precision_recall_dict}
        result_row = pd.DataFrame.from_dict(metrics_dict)
        results.append(result_row)
    session_result_df = pd.concat(results, ignore_index=True)
    mean_result = session_result_df['macro-avg F1-score'].mean(axis=0)
    print(f'Macro Avg F1-score: {mean_result}')
    if save:
        num_cols = session_result_df.select_dtypes(include=['number']).columns
        session_result_df[num_cols] = session_result_df[num_cols].round(3)
        session_result_df.to_csv(session_result_path, index=False)

if __name__ == '__main__':
    if len(sys.argv) == 4:
        chosen_label = sys.argv[1]
        cluster = sys.argv[2] # -1 for no clustering
        session = sys.argv[3] # 'all' for all sessions
    else:
        print(f'Format: {sys.argv[0]} <chosen_label> <cluster> <session>')
        sys.exit(1)
    if cluster == '-1':
        cluster_feature = ''
        cluster_num = -1
    else:
        cluster_feature = cluster.split(':')[0]
        cluster_num = int(cluster.split(':')[1])
    print('='*100);print(f'Chosen Label: {chosen_label}, Cluster Feature: {cluster_feature}, Cluster Num: {cluster_num}, Session: {session}');print('='*100)
    config = load_config(os.path.join(config_folder, 'xgboost.yaml'))
    compute_per_session_performance(chosen_label, cluster_feature, cluster_num, session, model_name='xgboost', config=config, save=True)
    # breakpoint()