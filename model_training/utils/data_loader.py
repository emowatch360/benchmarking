import numpy as np
from sklearn.utils import shuffle
from collections import Counter
import torch
from torch.utils.data import TensorDataset

class CommonDataLoader:
    def __init__(self, data_path, cv_type, num_class, session_filter='all',
                 label_type = 'discrete', feature_type_list = ['all'],
                 require_label_normalization=False, zscore_path='', require_shuffle=True, random_seed=42):
        self.cv_type = cv_type
        self.num_class = num_class
        self.label_type = label_type
        self.require_label_normalization = require_label_normalization
        self.data = self.load_data(data_path)
        self.feature_names = self.data['feature_names']
        self.session_labels = self.data['session_labels']
        if session_filter == 'all':
            self.num_samples = self.data['features'].shape[0]
            self.features = np.empty((self.num_samples, 0))
            self.non_temporal_features = np.empty((self.num_samples, 0)) # only needed for 2D features
            self.current_features = self.data['features']
            self.time_features = self.data['time_indices']
            self.prev_label_features = self.data['prev_survey_values']
            self.labels = self.data['labels']
            self.label_values = self.data['values']
            self.pid_labels = self.data['pid_labels']
            self.day_labels = self.data['day_labels']
            self.dg_labels = self.data['dg_labels']
            if 'sleep_statistics' in self.data.keys():
                self.sleep_stats = self.data['sleep_statistics']
            else:
                self.sleep_stats = np.empty((self.num_samples, 0))
            if 'sleep_features' in self.data.keys():
                self.sleep_features = self.data['sleep_features']
            else:
                self.sleep_features = np.empty((self.num_samples, 0))
        else:
            session_mask = self.session_labels == session_filter
            self.num_samples = self.data['features'][session_mask].shape[0]
            self.features = np.empty((self.num_samples, 0))
            self.non_temporal_features = np.empty((self.num_samples, 0)) # only needed for 2D features
            self.session_labels = self.session_labels[session_mask]
            self.current_features = self.data['features'][session_mask]
            self.time_features = self.data['time_indices'][session_mask]
            self.prev_label_features = self.data['prev_survey_values'][session_mask]
            self.labels = self.data['labels'][session_mask]
            self.label_values = self.data['values'][session_mask]
            self.pid_labels = self.data['pid_labels'][session_mask]
            self.day_labels = self.data['day_labels'][session_mask]
            self.dg_labels = self.data['dg_labels'][session_mask]
            if 'sleep_statistics' in self.data.keys():
                self.sleep_stats = self.data['sleep_statistics'][session_mask]
            else:
                self.sleep_stats = np.empty((self.num_samples, 0))
            if 'sleep_features' in self.data.keys():
                self.sleep_features = self.data['sleep_features'][session_mask]
            else:
                self.sleep_features = np.empty((self.num_samples, 0))
        if require_shuffle:
            assert session_filter == 'all', 'Shuffling is only supported for all sessions!'
            self.shuffle_data(random_seed)
        if self.require_label_normalization: # per user
            self.normalize_labels_per_user()
        elif self.label_type == 'continuous': # not it can be 'continuous' and require label normalization
            self.generate_continuous_labels()
        if cv_type in ['loso', 'groupkfold']:
            self.group_labels = self.pid_labels
        elif cv_type in ['lodo', 'time-series', 'time-series-custom', 'walk-forward']:
            self.group_labels = self.day_labels
        else:
            raise ValueError(f'Incorrect cv_type: {cv_type}')
        self.feature_type_dict = {
            # 'ppg': ['Respiration (breaths/min)', 'Heart Rate (bpm)', 'Spo2', 'Body Battery (%)'],
            'ppg': ['Respiration (breaths/min)', 'Heart Rate (bpm)', 'Spo2'],
            'skin-temperature': ['Temperature (celsius)'],
            'acc': ['Motion Intensity', 'Step Count', 'Total Count', 'Zero Crossing Count'],
            'time-hrv': ['SDNN', 'RMSSD', 'SDRMSSD', 'TINN'],
            'freq-hrv': ['VLF', 'LF', 'VHF', 'LFHF', 'LFn'],
            'non-linear-hrv': ['PIP', 'PAS', 'GI', 'PI', 'C1d', 'C2d', 'ApEn', 'MSEn', 'CMSEn', 'RCMSEn', 'HFD']
        }
        self.feature_indices = self.required_feature_indices(feature_type_list)
        self.zscore_path = zscore_path
    
    def shuffle_data(self, random_seed):
        (
        self.current_features,
        self.time_features,
        self.prev_label_features,
        self.labels,
        self.label_values,
        self.pid_labels,
        self.day_labels,
        self.dg_labels,
        self.sleep_stats,
        self.sleep_features,
        self.session_labels
        ) = shuffle(
            self.current_features,
            self.time_features,
            self.prev_label_features,
            self.labels,
            self.label_values,
            self.pid_labels,
            self.day_labels,
            self.dg_labels,
            self.sleep_stats,
            self.sleep_features,
            self.session_labels,
            random_state=random_seed
        )
    
    def load_data(self, data_path):
        return np.load(data_path)
    
    def required_feature_indices(self, feature_type_list):
        if self.current_features.ndim != 3: # feature indices selection only works for 2D features for now
            feature_indices = np.arange(self.current_features.shape[1])
        elif feature_type_list == ['all']:
            feature_indices = np.arange(self.current_features.shape[2])
        else:
            feature_to_index_dict = {}
            for i, feature in enumerate(self.feature_names):
                feature_to_index_dict[feature]=i
            feature_indices = []
            for feature_type in feature_type_list:
                # feature types are: ppg, skin-temperature, acc, time-hrv, freq-hrv, non-linear-hrv
                assert feature_type in self.feature_type_dict.keys(), f'Incorrect feature type: {feature_type}'
                features_of_interest = self.feature_type_dict[feature_type]
                feature_indices += [feature_to_index_dict[feature] for feature in features_of_interest]
        return feature_indices

    def normalize_labels_per_user(self):
        # for each user, identify the mean label value, and assign binary labels accordingly
        assert self.cv_type == 'loso', 'Label normalization per user is only supported for LOSO CV!'
        unique_pids = np.unique(self.pid_labels)
        for pid in unique_pids:
            pid_mask = self.pid_labels == pid
            mean_label_value = np.mean(self.label_values[pid_mask])
            self.labels[pid_mask] = (self.label_values[pid_mask] >= mean_label_value).astype(int)
    
    def generate_continuous_labels(self):
        # use the label values as continuous labels to create probabilities between 0 and 1
        # label values are always from 0 to 100
        # create continuous labels as a 2D array of shape (num_samples, 2)
        assert self.num_class == 2, 'Number of classes exceeds two!'
        self.labels = np.empty((self.num_samples, self.num_class), dtype=np.float32)
        self.labels[:, 1] = self.label_values / 100.0  # probability of class 1
        self.labels[:, 0] = 1.0 - self.labels[:, 1]    # probability of class 0
    
    def retrieve_2d_features(self,
                          include_curr_features=True,
                          include_time_index=False,
                          include_demo_info=False,
                          include_sleep_stats=False,
                          include_sleep_features=False,
                          include_prev_label=False,
                          as_tensor=False):
        '''
        Return 3 different feature matrices --
        (1) current features
        (2) sleep features
        (3) non-temporal features (demographics, time, sleep statistics)
        '''
        if include_time_index:
            self.non_temporal_features = np.hstack((self.non_temporal_features, self.time_features))
        if include_demo_info:
            self.non_temporal_features = np.hstack((self.non_temporal_features, self.dg_labels))
        if include_sleep_stats:
            self.non_temporal_features = np.hstack((self.non_temporal_features, self.sleep_stats))
        if include_prev_label:
            self.non_temporal_features = np.hstack((self.non_temporal_features, self.prev_label_features))
        if not include_curr_features:
            self.current_features = torch.tensor(np.empty((self.num_samples, 0)), dtype=torch.float32)
        else:
            self.current_features = self.current_features[:, :, self.feature_indices]
        if not include_sleep_features:
            self.sleep_features = torch.tensor(np.empty((self.num_samples, 0)), dtype=torch.float32)
        else:
            self.sleep_features = self.sleep_features[:, :, self.feature_indices]
        if as_tensor:
            if self.label_type == 'discrete':
                tensor_dataset = TensorDataset(torch.tensor(self.current_features, dtype=torch.float32),
                                            torch.tensor(self.sleep_features, dtype=torch.float32),
                                            torch.tensor(self.non_temporal_features, dtype=torch.float32),
                                            torch.tensor(self.labels, dtype=torch.long))
            elif self.label_type == 'continuous':
                tensor_dataset = TensorDataset(torch.tensor(self.current_features, dtype=torch.float32),
                                            torch.tensor(self.sleep_features, dtype=torch.float32),
                                            torch.tensor(self.non_temporal_features, dtype=torch.float32),
                                               torch.tensor(self.labels, dtype=torch.float32))
            else:
                raise ValueError(f'Incorrect label type: {self.label_type}')
            return tensor_dataset, self.group_labels
        return [self.current_features, self.sleep_features, self.non_temporal_features], self.group_labels

    def compute_mean_std_for_multimodal_data(self,
                                             include_time_index=False,
                                             include_demo_info=False,
                                             include_sleep_stats=False,
                                             include_prev_label=False):
                                             
        zscores = np.load(self.zscore_path)
        # Normalization parameters - Tuple containing mean and std, used for both pretraining and finetuning
        norm_mean = torch.FloatTensor(zscores['non_sleep_mean'][self.feature_indices])
        norm_std = torch.FloatTensor(zscores['non_sleep_std'][self.feature_indices])
        # Sleep Normalization parameters
        sleep_norm_mean = torch.FloatTensor(zscores['sleep_mean'][self.feature_indices])
        sleep_norm_std = torch.FloatTensor(zscores['sleep_std'][self.feature_indices])
        # Non Temporal Parameters (including time index, demographic, and sleep statistics)
        non_temporal_mean_list = []
        non_temporal_std_list = []
        if include_time_index:
            non_temporal_mean_list.append(zscores['time_mean'])
            non_temporal_std_list.append(zscores['time_std'])
        if include_demo_info:
            non_temporal_mean_list.append(zscores['dg_mean'])
            non_temporal_std_list.append(zscores['dg_std'])
        if include_sleep_stats:
            non_temporal_mean_list.append(zscores['sleep_stat_mean'])
            non_temporal_std_list.append(zscores['sleep_stat_std'])

        if non_temporal_mean_list:
            non_temporal_mean = torch.FloatTensor(np.concatenate(non_temporal_mean_list))
            non_temporal_std = torch.FloatTensor(np.concatenate(non_temporal_std_list))
        else:
            non_temporal_mean = torch.FloatTensor([])
            non_temporal_std = torch.FloatTensor([])

        multimodal_mean = (norm_mean, sleep_norm_mean, non_temporal_mean)
        multimodal_std = (norm_std, sleep_norm_std, non_temporal_std)
        return multimodal_mean, multimodal_std
        
        


    def retrieve_1d_features(self,
                          include_curr_features=True,
                          include_time_index=False,
                          include_demo_info=False,
                          include_sleep_stats=False,
                          include_sleep_features=False,
                          include_prev_label=False,
                          as_tensor=False):
        if include_curr_features:
            assert not np.any(np.isnan(self.current_features)), 'Some values are NaN!'
            self.features = np.hstack((self.features, self.current_features))
        if include_time_index:
            self.features = np.hstack((self.features, self.time_features))
        if include_demo_info:
            self.features = np.hstack((self.features, self.dg_labels))
        if include_sleep_stats:
            self.features = np.hstack((self.features, self.sleep_stats))
        if include_sleep_features:
            self.features = np.hstack((self.features, self.sleep_features))
        if include_prev_label:
            self.features = np.hstack((self.features, self.prev_label_features))
        assert self.features.shape[0] == self.num_samples
        # self.features = np.hstack((self.features, self.label_values.reshape(-1,1))) # debug: adding label as a feature
        # print statistics
        print(f'Data: {self.cv_type}, Feature Dim: {self.features.shape}, \
                Labels Split: {Counter(self.labels)}')
        if as_tensor:
            tensor_dataset = TensorDataset(torch.tensor(self.features, dtype=torch.float32),
                                           torch.tensor(self.labels, dtype=torch.long))
            return tensor_dataset, self.group_labels
        return self.features, self.group_labels
    
    def convert_scores_to_labels(self, scores, threshold):
        assert self.num_class == 2, 'Currently only works for two classes!'
        computed_labels = np.empty_like(scores)
        positive_map = scores >= threshold
        computed_labels[positive_map] = 1
        computed_labels[~positive_map] = 0
        return computed_labels

    def compute_personalized_labels(self, train_idx, valid_idx, test_idx):
        assert self.num_class == 2, 'Number of classes exceeds two!'
        # PID groups for train/valid/test
        train_pid_groups = self.pid_labels[train_idx]
        valid_pid_groups = self.pid_labels[valid_idx]
        test_pid_groups = self.pid_labels[test_idx]
        # label_values for train/valid/test -- take value from 0-100
        train_label_values = self.label_values[train_idx]
        print(train_label_values)
        valid_label_values = self.label_values[valid_idx]
        test_label_values = self.label_values[test_idx]
        # num unique PIDs
        n_unique_pids = np.unique(self.pid_labels).size
        # default threshold - 50
        threshold_per_pid_list = 50 * np.ones(n_unique_pids)
        # initialize label arrays
        train_labels = np.empty_like(train_idx)
        valid_labels = np.empty_like(valid_idx)
        test_labels = np.empty_like(test_idx)
        for pid_idx in np.arange(n_unique_pids):
            # print(pid_idx)
            # assuming pid_idx takes values 0 to n_unique_pids-1
            train_idx_per_pid = train_pid_groups==pid_idx
            # print(train_idx_per_pid)
            if np.any(train_idx_per_pid):
                threshold_per_pid_list[pid_idx] = np.mean(train_label_values[train_idx_per_pid])
            train_labels[train_idx_per_pid] = self.convert_scores_to_labels(train_label_values[train_idx_per_pid], 
                                                                            threshold_per_pid_list[pid_idx])
            valid_idx_per_pid = valid_pid_groups==pid_idx
            valid_labels[valid_idx_per_pid] = self.convert_scores_to_labels(valid_label_values[valid_idx_per_pid],
                                                                            threshold_per_pid_list[pid_idx])
            test_idx_per_pid = test_pid_groups==pid_idx
            test_labels[test_idx_per_pid] = self.convert_scores_to_labels(test_label_values[test_idx_per_pid],
                                                                            threshold_per_pid_list[pid_idx])
        print(f'All Thresholds: {threshold_per_pid_list}')
        return train_labels, valid_labels, test_labels

    def compute_labels(self, train_idx, valid_idx, test_idx):
        if self.cv_type in ['time-series-custom']:
            return self.compute_personalized_labels(train_idx, valid_idx, test_idx)
        else:
            return self.labels[train_idx], self.labels[valid_idx], self.labels[test_idx]

    def get_pid_labels(self, idxs):
        """
        Returns the PID labels for the given indices.
        """
        return self.pid_labels[idxs]
    
    def get_time_indices(self, idxs):
        """
        Returns the time indices for the given indices.
        """
        assert self.time_features.ndim == 2, 'Time features should be a 2D array!'
        return self.time_features[idxs][:,0]  # Assuming time_features is a 2D array with time indices in the first column

    def filter_test_indices_for_excluded_days(self, idxs, excluded_days=(1, 2)):
        """
        Removes test samples from fixed excluded days.
        """
        idxs = np.asarray(idxs)
        if idxs.size == 0:
            return idxs
        return idxs[~np.isin(self.day_labels[idxs], excluded_days)]

    def create_tensor_datasets_from_indices(self, tensor_dataset, group_labels, train_idx, valid_idx, test_idx,
                                            is_multimodal=False, print_log_path=None):
        if is_multimodal: 
            # This is for X1 = Physio data, X2 = Sleep data, X3 = Non-temporal data.
            # You may have to create copies of this function for different use cases.     
            X1, X2, X3, y = tensor_dataset.tensors
            group_labels = torch.Tensor(group_labels).long()
            train_set = TensorDataset(X1[train_idx], X2[train_idx], X3[train_idx], y[train_idx], group_labels[train_idx])
            val_set = TensorDataset(X1[valid_idx], X2[valid_idx], X3[valid_idx], y[valid_idx], group_labels[valid_idx])
            test_set = TensorDataset(X1[test_idx], X2[test_idx], X3[test_idx], y[test_idx], group_labels[test_idx])
        else:
            X, y = tensor_dataset.tensors
            train_set = TensorDataset(X[train_idx], y[train_idx])
            val_set = TensorDataset(X[valid_idx], y[valid_idx])
            test_set = TensorDataset(X[test_idx], y[test_idx])
        if self.label_type == 'discrete' and print_log_path is not None:
            with open(print_log_path, 'a') as log_file:
                log_file.write(f'Train Labels: {Counter(y[train_idx].numpy())}\n')
                log_file.write(f'Valid Labels: {Counter(y[valid_idx].numpy())}\n')
                log_file.write(f'Test Labels: {Counter(y[test_idx].numpy())}\n')
        return train_set, val_set, test_set

def build_round_robin_indices_per_user(users_for_train, k_per_user, batch_size):
    """
    Returns `ordered_indices` (dataset-local to the training split):
    Pack users_per_batch = batch_size // k_per_user users per batch,
    each with exactly k_per_user samples, in a round-robin fashion.
    users_for_train is a numpy array or list of user ids for each training example
    """
    users_per_batch = max(1, batch_size // k_per_user)
    # Map user -> list of dataset-local indices [0..len(train_idx)-1]
    user_to_indices = {}
    for local_i, u in enumerate(users_for_train):
        user_to_indices.setdefault(int(u), []).append(local_i)

    ordered = []
    working = {u: idxs[:] for u, idxs in user_to_indices.items()}
    users_seq = list(working.keys())
    while True:
        # Pick up to users_per_batch users that have >= k samples remaining
        candidates = [u for u in users_seq if len(working[u]) >= k_per_user]
        if len(candidates) < users_per_batch:
            break  # stop before a partial batch (drop_last behavior)
        selected = candidates[:users_per_batch]  # deterministic; or shuffle if desired
        for u in selected:
            take = working[u][:k_per_user]
            ordered.extend(take)
            working[u] = working[u][k_per_user:]
    return ordered
