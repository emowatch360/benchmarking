# combine pretraining datasets from two different dataset countries (e.g., 'jp' and 'sg')
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

    jp_load_path = os.path.join(
        data_folder,
        f'{num_class}class',
        f'jp_features{feature_dim}_epoch{context_min}',
        f'jp_{filename}'
    )
    sg_load_path = os.path.join(
        data_folder,
        f'{num_class}class',
        f'sg_features{feature_dim}_epoch{context_min}',
        f'sg_{filename}'
    )

    jp_npz = np.load(jp_load_path)
    sg_npz = np.load(sg_load_path)
    jp_data = {k: jp_npz[k].copy() for k in jp_npz.files}
    sg_data = {k: sg_npz[k].copy() for k in sg_npz.files}
    jp_npz.close()
    sg_npz.close()

    combined_data = {}
    for key in jp_data.keys():
        print(f'Key: {key}')
        if key in ['feature_names', 'sleep_feature_names']:
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