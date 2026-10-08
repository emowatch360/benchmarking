# combine datasets from two different dataset countries (e.g., 'jp' and 'sg')
import os
import sys
import numpy as np
sys.path.append('../visualize_features')
from utils import return_good_participant_mask # type: ignore
# from ..model_training.configs.shared_config import data_folder # type: ignore

# =============== Parameters ================
# data_folder = data_folder + 'corrected_hrv_numpy_arrays'
# feature_type = '2D' #'1D-stat' #'2D' #'1D-stat'
# num_class = 2
# context_min = 30
# feature_dim = 29
# label_str = 'stressed_current' # 'valence_current', 'arousal_current', 'stressed_current'
# sample_threshold = 10 # minimum number of samples per participant to be considered 'good'
# label_frac = 0.2 # minimum fraction of each class for a participant to be considered 'good'
# filter_good_participants = False # whether to filter out participants with too few samples / poor label distribution
# feature_normalization = False # whether features are normalized per participant

def merge_and_save_datasets(
        data_folder: str, feature_type: str, num_class: int, context_min: int, feature_dim: int,
        label_str: str, sample_threshold: int, label_frac: float, include_sleep: bool,
        filter_good_participants: bool, feature_normalization: bool, save_path: str) -> None:

    if save_path is None:
        filter_str = '_filtered' if filter_good_participants else ''
        feature_norm_str = '_normalized' if feature_normalization else ''
        sleep_str = '_all_sleep_features' if include_sleep else ''
        save_path = os.path.join(data_folder, f'{num_class}class', f'both_features{feature_dim}_epoch{context_min}',
                            f'all_emowatch_{feature_type}_{label_str}{sleep_str}_finetuning{filter_str}{feature_norm_str}.npz')
        
    # ================ Load JP/SG data ================
    jp_load_path = os.path.join(data_folder, f'{num_class}class', f'jp_features{feature_dim}_epoch{context_min}',
                            f'all_emowatch_{feature_type}_{label_str}{sleep_str}_finetuning.npz')
    label_str = label_str if label_str != 'stressed_current' else 'stress_level'
    sg_load_path = os.path.join(data_folder, f'{num_class}class', f'sg_features{feature_dim}_epoch{context_min}',
                            f'all_emowatch_{feature_type}_{label_str}{sleep_str}_finetuning.npz')

    jp_npz = np.load(jp_load_path)
    sg_npz = np.load(sg_load_path)
    jp_data = {k: jp_npz[k].copy() for k in jp_npz.files}
    sg_data = {k: sg_npz[k].copy() for k in sg_npz.files}
    jp_npz.close()
    sg_npz.close()

    # ================ Perform Feature Normalization ================
    if feature_normalization:
        # JP data
        jp_features_all = jp_data['features']
        if include_sleep:
            jp_sleep_features_all = jp_data['sleep_features']
            jp_sleep_statistics_all = jp_data['sleep_statistics']
        jp_unq_str_pids = np.unique(jp_data['str_pid_labels'])
        for pid in jp_unq_str_pids:
            pid_mask = (jp_data['str_pid_labels'] == pid)
            # current features
            features_pid = jp_features_all[pid_mask, :]
            mu = np.mean(features_pid, axis=0, keepdims=True)
            std = np.std(features_pid, axis=0, keepdims=True) + 1e-6
            jp_features_all[pid_mask, :] = (features_pid - mu) / std
            if include_sleep:
                # sleep features
                sleep_features_pid = jp_sleep_features_all[pid_mask, :]
                mu_sleep = np.mean(sleep_features_pid, axis=0, keepdims=True)
                std_sleep = np.std(sleep_features_pid, axis=0, keepdims=True) + 1e-6
                jp_sleep_features_all[pid_mask, :] = (sleep_features_pid - mu_sleep) / std_sleep
                # sleep statistics
                sleep_statistics_pid = jp_sleep_statistics_all[pid_mask, :]
                mu_stat = np.mean(sleep_statistics_pid, axis=0, keepdims=True)
                std_stat = np.std(sleep_statistics_pid, axis=0, keepdims=True) + 1e-6
                jp_sleep_statistics_all[pid_mask, :] = (sleep_statistics_pid - mu_stat) / std_stat
        # SG data
        sg_features_all = sg_data['features']
        if include_sleep:
            sg_sleep_features_all = sg_data['sleep_features']
            sg_sleep_statistics_all = sg_data['sleep_statistics']

        sg_unq_str_pids = np.unique(sg_data['str_pid_labels'])
        for pid in sg_unq_str_pids:
            pid_mask = (sg_data['str_pid_labels'] == pid)
            # current features
            features_pid = sg_features_all[pid_mask, :]
            mu = np.mean(features_pid, axis=0, keepdims=True)
            std = np.std(features_pid, axis=0, keepdims=True) + 1e-6
            sg_features_all[pid_mask, :] = (features_pid - mu) / std
            if include_sleep:
                # sleep features
                sleep_features_pid = sg_sleep_features_all[pid_mask, :]
                mu_sleep = np.mean(sleep_features_pid, axis=0, keepdims=True)
                std_sleep = np.std(sleep_features_pid, axis=0, keepdims=True) + 1e-6
                sg_sleep_features_all[pid_mask, :] = (sleep_features_pid - mu_sleep) / std_sleep
                # sleep statistics
                sleep_statistics_pid = sg_sleep_statistics_all[pid_mask, :]
                mu_stat = np.mean(sleep_statistics_pid, axis=0, keepdims=True)
                std_stat = np.std(sleep_statistics_pid, axis=0, keepdims=True) + 1e-6
                sg_sleep_statistics_all[pid_mask, :] = (sleep_statistics_pid - mu_stat) / std_stat
        # save normalized data temporarily
        fn_jp_path = jp_load_path.replace('.npz', '_normalized.npz')
        fn_sg_path = sg_load_path.replace('.npz', '_normalized.npz')
        np.savez(fn_jp_path, **jp_data)
        np.savez(fn_sg_path, **sg_data)
    else:
        combined_data = {}
        for key in jp_data.keys():
            print(f'Key: {key}')
            # check if single dimensional
            if key in ['feature_names', 'sleep_feature_names', 'sleep_statistic_names']:
                assert np.all(jp_data[key] == sg_data[key])
                values = jp_data[key] 
            elif key != 'pid_labels':
                values = np.concatenate((jp_data[key], sg_data[key]), axis=0)
            else:
                assert np.min(jp_data[key]) == 1, 'JP PIDs should start from 1!'
                assert np.min(sg_data[key]) == 1, 'SG PIDs should start from 1!'
                max_jp_pid = np.max(jp_data[key])
                values = np.concatenate((jp_data[key], max_jp_pid + sg_data[key]), axis=0)
            combined_data[key] = values

        # ================ Filter good participants ================
        if filter_good_participants:
            labels = combined_data['labels'].reshape(-1,1)
            str_pids = combined_data['str_pid_labels'].reshape(-1,1)
            pid_mask = return_good_participant_mask(labels, str_pids, sample_threshold=sample_threshold,
                                                            label_frac=label_frac, num_class=num_class)
            for key in combined_data.keys():
                if key in ['feature_names', 'sleep_feature_names', 'sleep_statistic_names']:
                    continue
                if combined_data[key].ndim == 1:
                    combined_data[key] = combined_data[key][pid_mask]
                else:
                    combined_data[key] = combined_data[key][pid_mask, :]
        print(f'Combined data feature shapes: {combined_data["features"].shape}')
        # ================ Save combined data ================
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(save_path, **combined_data)