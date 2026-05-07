# combine pretraining datasets from two different dataset countries (e.g., 'c1' and 'c2')
import os
import numpy as np
from ..model_training.configs.shared_config import data_folder # type: ignore

def merge_and_save_pretrained_datasets(
        data_folder: str, num_class: int, context_min: int, feature_dim: int,
        include_sleep: bool, save_path: str = None) -> None:

    suffix = '_sleep' if include_sleep else ''
    filename = f'emowatch_pretraining{suffix}.npz'

    if save_path is None:
        save_path = os.path.join(
            data_folder,
            f'{num_class}class',
            f'both_features{feature_dim}_epoch{context_min}',
            f'both_{filename}'
        )

    c1_load_path = os.path.join(
        data_folder,
        f'{num_class}class',
        f'c1_features{feature_dim}_epoch{context_min}',
        f'c1_{filename}'
    )
    c2_load_path = os.path.join(
        data_folder,
        f'{num_class}class',
        f'c2_features{feature_dim}_epoch{context_min}',
        f'c2_{filename}'
    )

    c1_npz = np.load(c1_load_path)
    c2_npz = np.load(c2_load_path)
    c1_data = {k: c1_npz[k].copy() for k in c1_npz.files}
    c2_data = {k: c2_npz[k].copy() for k in c2_npz.files}
    c1_npz.close()
    c2_npz.close()

    combined_data = {}
    for key in c1_data.keys():
        print(f'Key: {key}')
        if key in ['feature_names', 'sleep_feature_names']:
            assert np.all(c1_data[key] == c2_data[key])
            values = c1_data[key]
        elif key != 'pid_labels':
            values = np.concatenate((c1_data[key], c2_data[key]), axis=0)
        else:
            assert np.min(c1_data[key]) == 1, 'C1 PIDs should start from 1!'
            assert np.min(c2_data[key]) == 1, 'C2 PIDs should start from 1!'
            max_c1_pid = np.max(c1_data[key])
            values = np.concatenate((c1_data[key], max_c1_pid + c2_data[key]), axis=0)
        combined_data[key] = values

    merged_key = 'sleep_features' if include_sleep else 'features'
    print(f'Combined pretraining data shape: {combined_data[merged_key].shape}')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, **combined_data)


if __name__ == '__main__':
    data_folder = data_folder + 'numpy_arrays_pretraining'
    num_class = 2
    context_min = 30
    feature_dim = 28
    include_sleep = False

    merge_and_save_pretrained_datasets(
        data_folder=data_folder,
        num_class=num_class,
        context_min=context_min,
        feature_dim=feature_dim,
        include_sleep=include_sleep,
        save_path=None
    )