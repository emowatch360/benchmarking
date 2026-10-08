import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold

class CrossValidator:
    def __init__(self, cv_type, valid_size=2, start_group_idx=11,
                 temporal_window=9, n_splits=5,
                 print_log_path=None):
        self.print_log_path = print_log_path
        self.cv_type = cv_type
        self.valid_size = valid_size
        if cv_type in ['loso', 'lodo']:
            self.cv = LeaveOneGroupOut()
        elif cv_type in ['groupkfold']: # group k-fold
            self.cv = GroupKFold(n_splits=n_splits)
        elif cv_type in ['time-series', 'time-series-custom']:
            self.start_group_idx = start_group_idx
        elif cv_type == 'walk-forward':
            self.start_group_idx = start_group_idx
            self.temporal_window = temporal_window
        else:
            raise ValueError(f'Unknown CV Type: {cv_type}')
    
    def logo_split(self, X, y, groups):
        """
        Aim is to generate splits where one of the groups is test while everything else is train/valid
        """
        for train_idx, test_idx in self.cv.split(X, y, groups):
            unique_train_groups = np.unique(groups[train_idx])
            assert unique_train_groups.size > self.valid_size, "Valid set required larger than train set!"
            # Select validation groups randomly from train groups
            valid_groups = np.random.choice(unique_train_groups, size=self.valid_size, replace=False)
            valid_idx = np.flatnonzero(np.isin(groups, valid_groups))
            # Ensure valid_idx does not overlap with test_idx
            train_idx = np.setdiff1d(train_idx, valid_idx)
            yield train_idx, valid_idx, test_idx
    
    def ts_split(self, X, y, groups):
        """
        Aim is to generate splits such that all train/valid indices precede test indices (to maintain a temporal order)
        """
        unique_groups = np.unique(groups)
        n_unique_groups = unique_groups.size
        assert self.start_group_idx > self.valid_size, "Start group index must be greater than valid size!"
        # assumes all group numberings are in order from 1 to n_unique_groups
        if np.any(groups==0):
            num_zero_groups = np.sum(groups==0)
            print(f'There are {num_zero_groups} in the dataset! Ignoring for now')
        assert (np.min(groups) == 0) or (np.min(groups) == 1) or (np.min(groups) == 2) # two for sleep, 0 exception is only for oac10 day0
        assert (np.max(groups) == n_unique_groups-1) or (np.max(groups) == n_unique_groups) or (np.max(groups) == n_unique_groups+1) # for sleep where day1 is missing
        for test_group in np.arange(self.start_group_idx, np.max(groups)+1):
            test_idx = np.flatnonzero(np.isin(groups, test_group))
            valid_groups = np.arange(test_group-self.valid_size, test_group)
            # valid_groups = np.random.choice(np.arange(test_group), size=self.valid_size, replace=False)
            valid_idx = np.flatnonzero(np.isin(groups, valid_groups))
            train_groups = np.setdiff1d(np.arange(test_group), valid_groups)
            train_idx = np.flatnonzero(np.isin(groups, train_groups))
            yield train_idx, valid_idx, test_idx
    
    # perform walk forward validation which is similar to ts_split, except that it does not require a start_group_idx
    def walk_forward_split(self, X, y, groups):
        """
        Aim is to generate splits such that all train/valid indices precede test indices (to maintain a temporal order)
        """
        unique_groups = np.unique(groups)
        n_unique_groups = unique_groups.size
        assert self.start_group_idx > self.valid_size, "Start group index must be greater than valid size!"
        # assumes all group numberings are in order from 1 to n_unique_groups
        if np.any(groups==0):
            num_zero_groups = np.sum(groups==0)
            print(f'There are {num_zero_groups} in the dataset! Ignoring for now')
        assert (np.min(groups) == 0) or (np.min(groups) == 1) or (np.min(groups) == 2) # two for sleep, 0 exception is only for oac10 day0
        assert (np.max(groups) == n_unique_groups-1) or (np.max(groups) == n_unique_groups) or (np.max(groups) == n_unique_groups+1) # for sleep where day1 is missing
        # assign the test group randomly instead of in order
        # sanity check to see if order matters
        # test_groups = np.random.choice(np.arange(self.start_group_idx, np.max(groups)+1), size=n_unique_groups - self.start_group_idx + 1, replace=False)
        # print(test_groups)
        for test_group in np.arange(self.start_group_idx, np.max(groups)+1):
            # for test_group in test_groups:
            test_idx = np.flatnonzero(np.isin(groups, test_group))
            valid_groups = np.arange(test_group-self.valid_size, test_group)
            # valid_groups = np.random.choice(np.arange(test_group), size=self.valid_size, replace=False)
            valid_idx = np.flatnonzero(np.isin(groups, valid_groups))
            assert test_group - self.temporal_window >= 0, "Test group must be greater than train/valid window size!"
            train_groups = np.arange(test_group-self.temporal_window, test_group-self.valid_size)
            train_idx = np.flatnonzero(np.isin(groups, train_groups))
            yield train_idx, valid_idx, test_idx

    def split(self, X, y, groups):
        """
        Generates train, validation, and test indices.
        Args:
        - X (numpy array): Feature matrix
        - y (numpy array): Labels
        - groups (numpy array): Group labels for each sample
        Yields:
        - train_idx, valid_idx, test_idx (numpy arrays): Indices for train, validation, and test sets
        """
        if self.cv_type in ['loso', 'lodo', 'groupkfold']:
            yield from self.logo_split(X, y, groups)
        elif self.cv_type in ['time-series', 'time-series-custom']:
            yield from self.ts_split(X, y, groups)
        elif self.cv_type == 'walk-forward':
            yield from self.walk_forward_split(X, y, groups)
    
    def perform_assertions_on_group_labels(self, group_labels, train_idx, val_idx, test_idx):
        """
        Ensure that there is not intersection between train/val/test indices based on group labels
        """
        train_groups = np.unique(group_labels[train_idx])
        val_groups = np.unique(group_labels[val_idx])
        test_groups = np.unique(group_labels[test_idx])
        if self.print_log_path is not None:
            with open(self.print_log_path, 'a') as log_file:
                log_file.write(f'Groups: Train:{train_groups} | 'f'Valid:{val_groups} | 'f'Test:{test_groups}\n')
        assert np.intersect1d(train_groups, val_groups).size == 0
        assert np.intersect1d(val_groups, test_groups).size == 0
        assert np.intersect1d(train_groups, test_groups).size == 0