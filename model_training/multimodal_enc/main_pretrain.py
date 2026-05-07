# main.py
import os
import sys
import yaml
import numpy as np # type: ignore
import torch
from torch.utils.data import DataLoader, Subset # type: ignore
import pandas as pd # type: ignore

from ..configs.shared_config import (data_folder, config_folder, seed) # type: ignore
from .models.conv1d import Conv1D_Encoder
from .models.resnet1d import ResNet1D_Encoder
from .models.mha1d import MultiHeadAttention1DEncoder
from .pretraining.pretrain import SSLPretrainer
from .pretraining.crossl_utils import CroSSLAggregator
from ..utils.common_utils import load_config
from ..utils.common_metrics import perform_zscore_normalization, perform_zscore_normalization_pretraining
from ..utils.reproduce import set_seed, get_dataloader_seed_components

cfg = load_config(os.path.join(config_folder, 'multimodal_enc.yaml'))

FEATURE_TYPE_DICT = {
    'ppg': ['Respiration (breaths/min)', 'Heart Rate (bpm)', 'Spo2'],
    'skin-temperature': ['Temperature (celsius)'],
    'acc': ['Motion Intensity', 'Step Count', 'Total Count', 'Zero Crossing Count'],
    'time-hrv': ['SDNN', 'RMSSD', 'SDRMSSD', 'TINN'],
    'freq-hrv': ['VLF', 'LF', 'VHF', 'LFHF', 'LFn'],
    'non-linear-hrv': ['PIP', 'PAS', 'GI', 'PI', 'C1d', 'C2d', 'ApEn', 'MSEn', 'CMSEn', 'RCMSEn', 'HFD']
}

def _feature_lookup_keys(feature_name):
    feature_name = str(feature_name)
    keys = [feature_name]
    if feature_name.startswith('sl-'):
        keys.append(feature_name[3:])
    return keys

def _required_feature_indices(feature_names, feature_axis_size, feature_type_list):
    if feature_type_list == ['all']:
        return np.arange(feature_axis_size)

    feature_to_index_dict = {}
    for i, feature_name in enumerate(feature_names):
        for key in _feature_lookup_keys(feature_name):
            feature_to_index_dict.setdefault(key, []).append(i)

    feature_indices = []
    for feature_type in feature_type_list:
        assert feature_type in FEATURE_TYPE_DICT, f'Incorrect feature type: {feature_type}'
        for feature in FEATURE_TYPE_DICT[feature_type]:
            assert feature in feature_to_index_dict, f'Missing feature: {feature}'
            feature_indices += feature_to_index_dict[feature]

    return np.array(feature_indices)

def _select_pretraining_features(pretrain_data_path, data_key, feature_type_list):
    with np.load(pretrain_data_path) as pretrain_npz:
        pretrain_data = pretrain_npz[data_key]
        feature_name_key = 'sleep_feature_names' if data_key == 'sleep_features' else 'feature_names'
        if feature_name_key not in pretrain_npz.files:
            feature_name_key = 'feature_names'
        feature_names = pretrain_npz[feature_name_key]

    assert pretrain_data.ndim == 3, f'{data_key} must be 3D with shape (samples, time, features)!'
    feature_axis = 2
    assert len(feature_names) == pretrain_data.shape[feature_axis], \
        f'{feature_name_key} does not match {data_key} feature dimension!'
    original_feature_count = pretrain_data.shape[feature_axis]
    feature_indices = _required_feature_indices(
        feature_names, original_feature_count, feature_type_list)
    pretrain_data = np.take(pretrain_data, feature_indices, axis=feature_axis)
    return pretrain_data, feature_indices, original_feature_count

def _select_norm_params(norm_params, feature_indices, original_feature_count, selected_feature_count):
    norm_mean, norm_std = norm_params
    if norm_mean.shape[0] == selected_feature_count:
        return norm_params
    assert norm_mean.shape[0] == original_feature_count, \
        f'Normalization mean has {norm_mean.shape[0]} features; expected {selected_feature_count} or {original_feature_count}.'
    assert norm_std.shape[0] == original_feature_count, \
        f'Normalization std has {norm_std.shape[0]} features; expected {original_feature_count}.'
    feature_indices_tensor = torch.as_tensor(feature_indices, dtype=torch.long, device=norm_mean.device)
    return norm_mean[feature_indices_tensor], norm_std[feature_indices_tensor.to(norm_std.device)]

def perform_ssl_pretraining(pretrain_data_path, norm_params, 
                            model_name='multimodal_enc', data_type='cp', config=None, save=False):
    
    # DataLoader seed setup
    type_seed = seed if data_type == 'cp' else seed + 1
    set_seed(type_seed)
    _, generator = get_dataloader_seed_components(type_seed)
    
    assert data_type in ['cp', 'sp'], "data_type must be either 'cp' or 'sp'"
    if config:
        cfg = config
    else:
        cfg = load_config(os.path.join(config_folder, f'{model_name}.yaml'))
    pretraining_folder = config_folder + '/../multimodal_enc/pretraining/'
    use_ssl_pretraining = cfg['training'][f'use_{data_type}_ssl_pretraining']
    use_pretrained_encoders = cfg['training']['use_pretrained_encoders']
    pretraining_type = cfg['pretraining']['pretraining_type']
    model_type = cfg['model']['model_type']
    conv_layers = cfg['model']['conv_layers'] if data_type == 'cp' else cfg['model']['sp_conv_layers']
    batch_size = cfg['pretraining']['batch_size']
    epoch = cfg['pretraining']['epochs']
    feature_type_list = cfg['data']['feature_type_list']
    feature_tag = 'all' if feature_type_list == ['all'] else '-'.join(sorted(feature_type_list))

    pretrained_encoder = [] # If use_cp_ssl_pretraining is False, pretrained_cp_encoder will be None. In train.py and multimodal encoder, it will skip the appropriate SSL code if it is None.
    pretrained_aggregator = None

    # SSL Pretraining for Current Physiological Data
    if not use_ssl_pretraining:
        return [], None

    # Build the path for encoder 0 as a proxy — if it exists, all encoders exist
    enc_check_path = os.path.join(pretraining_folder, 
        f'encoder_{data_type}_{pretraining_type}_{model_type}{conv_layers}_batch{batch_size}_epoch{epoch}_feat_{feature_tag}_0_pretrained.pt')
    encoders_already_saved = os.path.exists(enc_check_path)
    # If there are pretrained encoders to use:
    if encoders_already_saved or use_pretrained_encoders:
        # Pretrained encoder should be present in the pretraining subfolder, with .pt extension.
        # Naming convention: "encoder_cp_{i}_pretrained" for i in [1, 2, ..., len(feature_split)]
        # MAKE SURE THAT THE MODEL TYPE SPECIFIED IN multimodal_enc.yaml for pretraining matches that of the encoder being loaded.
        for i in range(0, len(cfg['model']['feature_split'])):
            enc_path = os.path.join(pretraining_folder, f'encoder_{data_type}_{pretraining_type}_{model_type}{conv_layers}_batch{batch_size}_epoch{epoch}_feat_{feature_tag}_{i}_pretrained.pt')
            enc_checkpoint = torch.load(enc_path)
            if 'logit_scale' in enc_checkpoint:
                logit_scale_value = enc_checkpoint['logit_scale']
            config = enc_checkpoint['encoder_config']
            if cfg['model']['model_type'].upper() == 'CNN':
                encoder = Conv1D_Encoder(input_dim=config['input_features'], conv_layers=config['conv_layers'], dropout=config['dropout'])
            elif cfg['model']['model_type'].upper() == 'RESNET':
                encoder = ResNet1D_Encoder(input_dim=config['input_features'], base_channels=config['conv_layers'][-1], dropout=config['dropout'])
            elif cfg['model']['model_type'].upper() == 'MHA':
                encoder = MultiHeadAttention1DEncoder(input_dim=config['input_features'], embed_dim=config['conv_layers'][-1], num_heads=config['num_heads'], dropout=config['dropout'])
            else:
                # Default to CNN
                encoder = Conv1D_Encoder(input_dim=config['input_features'], conv_layers=config['conv_layers'], dropout=config['dropout'])

            encoder.load_state_dict(enc_checkpoint['encoder_state_dict'])
            encoder.eval()
            encoder.to(cfg['training']['device'])
            pretrained_encoder.append(encoder)
            
        # aggregator
        if pretraining_type == 'regularization-vicreg':
            agg_path = os.path.join(
            pretraining_folder,
            f"aggregator_{data_type}_{pretraining_type}_{model_type}{conv_layers}_batch{batch_size}_epoch{epoch}_feat_{feature_tag}_pretrained.pt")
            agg_checkpoint = torch.load(agg_path)
            agg_config = agg_checkpoint['aggregator_config']
            pretrained_aggregator = CroSSLAggregator(
                num_feature_groups=agg_config['num_feature_groups'],
                embedding_dim=agg_config['embedding_dim'],
            )
            pretrained_aggregator.load_state_dict(agg_checkpoint['aggregator_state_dict'])
            pretrained_aggregator.eval()
            pretrained_aggregator.to(cfg['training']['device'])

    # If there are no pretrained encoders to use, pretrain from scratch.
    else:
        # Load and normalize pretraining data
        data_key = 'features' if data_type == 'cp' else 'sleep_features'
        pretrain_data, feature_indices, original_feature_count = _select_pretraining_features(
            pretrain_data_path, data_key, feature_type_list)
        pretrain_set = torch.FloatTensor(pretrain_data)
        pretrain_mean, pretrain_std = _select_norm_params(
            norm_params, feature_indices, original_feature_count=original_feature_count,
            selected_feature_count=pretrain_data.shape[-1])
        pretrain_set = perform_zscore_normalization_pretraining(pretrain_set, mean=pretrain_mean, std=pretrain_std)
        pretrain_set = pretrain_set[torch.randperm(pretrain_set.shape[0], generator=generator)] # Random shuffle
        # Perform encoder pretraining
        ssl_pretrainer_obj = SSLPretrainer(cfg, pretrain_set, data_type)
        ssl_pretrainer_obj.train()
        pretrained_encoder = ssl_pretrainer_obj.get_encoders()
        for encoder in pretrained_encoder:
            encoder.eval()
        if pretraining_type == 'regularization-vicreg':
            pretrained_aggregator = ssl_pretrainer_obj.get_aggregator()
            pretrained_aggregator.eval()
        if save:
            ssl_pretrainer_obj.save_encoders() # also saves aggregator if present
    return pretrained_encoder, pretrained_aggregator
