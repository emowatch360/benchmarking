import yaml
import os
import torch

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def convert_categrical_to_zero_indexed(curr_groups: torch.Tensor, all_groups: list) -> torch.Tensor:
    # sort all_groups to ensure consistent mapping
    sorted_groups = sorted(all_groups)
    group_to_index = {group: idx for idx, group in enumerate(sorted_groups)}
    mapped_indices = [group_to_index[group] for group in curr_groups.cpu().tolist()]
    mapped_indices = torch.LongTensor(mapped_indices).to(curr_groups.device)
    return mapped_indices
    
# def create_data_and_save_paths(model_name, config, data_folder, results_folder, label, algorithm, change_lr=False):
def create_data_and_save_paths(model_name, config, data_folder, results_folder,
                               label, algorithm='', dg_model='', change_lr=False,
                               curr_time=''):
    dataset_name = config['data']['name']
    feature_dim = config['data']['feature_dim']
    include_sleep = config['data']['use_sleep_path'] or config['data']['include_sleep_features'] or config['data']['include_sleep_stats']
    cv_type = config['data']['cv']
    curr_features = 'c1' if config['data']['include_curr_features'] else 'c0'
    prev_label = 'p1' if config['data']['include_prev_label'] else 'p0'
    time_index = 't1' if config['data']['include_time_index'] else 't0'
    dg_info = 'd1' if config['data']['include_demo_info'] else 'd0'
    sleep_stat = 'ss1' if config['data']['include_sleep_stats'] else 'ss0'
    sleep_feat = 'sf1' if config['data']['include_sleep_features'] else 'sf0'
    # require_ln = 'ln1' if config['data']['require_label_normalization'] else 'ln0'
    participant_normalization=config['data']['participant_normalization']
    sleep_normalization=config['data']['sleep_normalization']
    assert participant_normalization==False, "Participant normalization should be set to False"
    if include_sleep:
        assert sleep_normalization==False, "Sleep normalization should be set to False when including sleep features"
        data_path = os.path.join(data_folder, f'all_{dataset_name}_{feature_dim}_{label}_all_sleep_features_finetuning.npz')
    elif sleep_normalization:
        assert include_sleep==False, "Include sleep features should be set to False when using sleep normalization"
        data_path = os.path.join(data_folder, f'all_{dataset_name}_{feature_dim}_{label}_finetuning_sleep_normalized.npz')
    else:
        data_path = os.path.join(data_folder, f'all_{dataset_name}_{feature_dim}_{label}_finetuning.npz') # modified!! #all_sleep_features_
    if model_name != 'xgboost':
        # if participant_normalization:
            # data_path = data_path.replace('.npz', '_normalized.npz')
        # results_fname = f'{label}_{feature_dim}_{curr_time}_all.csv'
        results_fname = f'{label}_{dg_model}_layers{"-".join(map(str, config["model"]["conv_layers"]))}_{curr_features}_{prev_label}_{time_index}_{dg_info}_{sleep_stat}_{sleep_feat}_sn{int(sleep_normalization)}_pn{int(participant_normalization)}.csv'
    else:
        md = config['training']['max_depth']
        mcw = config['training']['min_child_weight']
        gs = config['training']['gamma']
        ss = config['training']['subsample']
        csb = config['training']['colsample_bytree']
        ra = config['training']['reg_alpha']
        rl = config['training']['reg_lambda']
        results_fname = f'{label}_{feature_dim}_md{md}_mcw{mcw}_gs{gs}_ss{ss}_csb{csb}_ra{ra}_rl{rl}.csv'
        # results_fname = f'{label}_{curr_features}_{prev_label}_{time_index}_{dg_info}_{sleep_stat}_{sleep_feat}_sn{int(sleep_normalization)}_pn{int(participant_normalization)}.csv'
    all_result_path = os.path.join(results_folder, results_fname)
    return data_path, all_result_path