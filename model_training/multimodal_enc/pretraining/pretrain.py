# pretrain.py
import os
import sys
sys.path.append('../../utils/')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from ..models.conv1d import Conv1D_Encoder
from ..models.resnet1d import ResNet1D_Encoder
from ..models.mha1d import MultiHeadAttention1DEncoder
from .crossl_utils import CroSSLAggregator, get_random_masks, vicreg_loss
from ...utils.common_metrics import compute_accuracy
from ...utils.reproduce import set_seed, get_dataloader_seed_components
from ...configs.shared_config import (data_folder, config_folder, seed) # type: ignore

class SSLPretrainer:
    def __init__(self, cfg, chosen_label, data, data_type='cp'):

        # SSL Configs
        self.cfg = cfg
        self.chosen_label = chosen_label
        self.data_type = data_type
        self.feature_split = cfg['model']['feature_split']  # List that shows the data split
        self.pretrain_model = cfg['model']['model_type']
        
        if data_type == 'cp':
            self.conv_layers = cfg['model']['conv_layers']
        elif data_type == 'sp':
            self.conv_layers = cfg['model']['sp_conv_layers']
        else: 
            self.conv_layers = cfg['model']['conv_layers']

        self.pretraining_type=cfg['pretraining']['pretraining_type']
        self.input_dim = cfg['pretraining']['input_dim']
        self.epochs=cfg['pretraining']['epochs']
        self.batch_size=cfg['pretraining']['batch_size']
        self.lr=cfg['pretraining']['learning_rate']
        self.dropout = cfg['pretraining']['dropout']
        self.device = cfg['pretraining']['device']
        self.use_scheduler=cfg['pretraining']['use_scheduler']
        # crossl specific configs
        self.masking = cfg['pretraining']['masking']
        self.coverage = cfg['pretraining']['coverage']
        self.sim_coeff = cfg['pretraining']['sim_coeff']
        self.std_coeff = cfg['pretraining']['std_coeff']
        self.cov_coeff = cfg['pretraining']['cov_coeff']
        self.std_const = cfg['pretraining']['std_const']
        
        feature_type_list = cfg['data']['feature_type_list']
        self.feature_tag = 'all' if feature_type_list == ['all'] else '-'.join(sorted(feature_type_list))


        self.patience=cfg['early_stopping']['patience']
        self.delta=cfg['early_stopping']['delta']
        
        self.logit_scale = nn.Parameter(torch.zeros(1, device=self.device)) # exp(0) = 1.0 initially
        self.scale_optimizer = None  

        # Data shape is N x T x F (Number of samples x Time steps x Features): Should be (38385 + 19019) x 30 x 28 at the moment
        self.data = data

        # Split should add up to the number of features
        assert sum(self.feature_split) == data.shape[2], f"Feature split {self.feature_split} sum does not add up to {data.shape[2]}"

        # Initialize data and encoders
        self.split_data = self._split_features()
        self.encoders = self._initialize_encoders(self.pretrain_model)
        self.embedding_dim = self._get_embedding_dim()
        self.aggregator = self._initialize_aggregator()

    def _split_features(self):
        split_data = []
        start_idx = 0
        for split_size in self.feature_split:
            end_idx = start_idx + split_size
            split_data.append(self.data[:, :, start_idx:end_idx])
            start_idx = end_idx
        return split_data

    def _initialize_encoders(self, pretrain_model):
        encoders = []
        for i, split_size in enumerate(self.feature_split):
            if pretrain_model.upper() == 'CNN':
                encoder = Conv1D_Encoder(input_dim=split_size, conv_layers=self.conv_layers, dropout=self.dropout)
                encoders.append(encoder.to(self.device))
            if pretrain_model.upper() == 'RESNET':
                encoder = ResNet1D_Encoder(input_dim=split_size, base_channels=self.conv_layers[-1], dropout=self.dropout)
                encoders.append(encoder.to(self.device))
            if pretrain_model.upper() == 'MHA':
                encoder = MultiHeadAttention1DEncoder(input_dim=split_size, embed_dim=self.conv_layers[-1], num_heads=4, dropout=self.dropout)
                encoders.append(encoder.to(self.device))
        return encoders

    def _get_embedding_dim(self):
        if self.pretrain_model.upper() in ['CNN', 'RESNET', 'MHA']:
            return self.conv_layers[-1]
        raise ValueError(f"Unsupported pretraining model type: {self.pretrain_model}")

    def _initialize_aggregator(self):
        return CroSSLAggregator(
            num_feature_groups=len(self.feature_split),
            embedding_dim=self.embedding_dim,
        ).to(self.device)

    def train(self):

        optimizers = [optim.Adam(encoder.parameters(), lr=self.lr) for encoder in self.encoders]
        aggregator_optimizer = None
        if self.pretraining_type in ['contrastive-pairwise', 'contrastive-loo']:
            self.scale_optimizer = optim.Adam([self.logit_scale], lr=self.lr)
        if self.pretraining_type == 'regularization-vicreg':
            aggregator_optimizer = optim.Adam(self.aggregator.parameters(), lr=self.lr)
        
        num_samples = self.data.shape[0]
        num_batches = (num_samples + self.batch_size - 1) // self.batch_size
        num_feature_groups = len(self.encoders)
        
        self.split_data = [torch.FloatTensor(d).to(self.device) for d in self.split_data]
        
        rng = np.random.RandomState(seed)
        g = torch.Generator(device=self.device)
        g.manual_seed(seed)
        for epoch in range(1, self.epochs + 1):
            total_loss = 0
            indices = rng.permutation(num_samples)
            
            for encoder in self.encoders:
                encoder.train()
            if aggregator_optimizer is not None:
                self.aggregator.train()
            
            for batch_idx in tqdm(range(num_batches), desc=f"Epoch {epoch}/{self.epochs}"):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]
                current_batch_size = len(batch_indices)
                
                embeddings = []
                for i, encoder in enumerate(self.encoders):
                    batch_data = batch_data = self.split_data[i][batch_indices]
                    embedding = encoder(batch_data)
                    if self.pretraining_type != 'regularization-vicreg':
                        embedding = F.normalize(embedding, dim=1)
                    embeddings.append(embedding)
                
                # NT-Xent Loss - Used in SimCLR and SigRep
                if self.pretraining_type == "contrastive-ntxent":
                    loss = 0.
                    temperature = 0.5
                    num_pairs = 0
                    for i in range(num_feature_groups):
                        for j in range(i + 1, num_feature_groups):
                            B = embeddings[i].size(0)
                            z = torch.cat([embeddings[i], embeddings[j]], dim=0)
                            sim = torch.mm(z, z.T) / temperature
                            mask = torch.eye(2 * B, device=self.device).bool()
                            sim.masked_fill_(mask, float('-inf'))
                            labels = torch.cat([
                                torch.arange(B, 2 * B, device=self.device),
                                torch.arange(0, B, device=self.device)
                            ])
                            loss += F.cross_entropy(sim, labels)
                            num_pairs += 1
                    loss /= num_pairs
                
                # Choose between pairwise or leave-one-out contrastive learning
                # Loss calculation taken from SleepFM Code:
                # https://github.com/zou-group/sleepfm-clinical/blob/main/sleepfm/pipeline/pretrain.py#L51-L109
                elif self.pretraining_type == "contrastive-pairwise":
                    loss = 0.
                    num_pairs = 0
                    for i in range(num_feature_groups):
                        for j in range(i + 1, num_feature_groups):
                            logits = torch.matmul(embeddings[i], embeddings[j].transpose(0, 1)) * self.logit_scale.exp()
                            labels = torch.arange(logits.shape[0], device=self.device)
                            
                            l = F.cross_entropy(logits, labels, reduction="sum")
                            loss += l
                            
                            l = F.cross_entropy(logits.transpose(0, 1), labels, reduction="sum")
                            loss += l
                            
                            num_pairs += 2
                    loss /= num_pairs  # divide by number of pair directions first

                    # apply batch normalization after computing total loss
                    loss /= current_batch_size
                    
                elif self.pretraining_type == "contrastive-loo":
                    loss = 0.
                    for i in range(num_feature_groups):
                        other_emb = torch.stack([embeddings[j] for j in list(range(i)) + list(range(i + 1, num_feature_groups))]).sum(0) / (num_feature_groups - 1)
                        logits = torch.matmul(embeddings[i], other_emb.transpose(0, 1)) * self.logit_scale.exp()
                        labels = torch.arange(logits.shape[0], device=self.device)
                        
                        l = F.cross_entropy(logits, labels, reduction="sum")
                        loss += l
                        
                        l = F.cross_entropy(logits.transpose(0, 1), labels, reduction="sum")
                        loss += l
                    loss /= (num_feature_groups * 2)

                    # apply batch normalization after computing total loss
                    loss /= current_batch_size

                # Regularization-based Approach (e.g. CroSSL VICReg)
                elif self.pretraining_type == 'regularization-vicreg':
                    modality_embeddings = torch.stack(embeddings, dim=1)
                    mask_a, mask_b = get_random_masks(
                        batch_size=current_batch_size,
                        num_feature_groups=num_feature_groups,
                        embedding_dim=modality_embeddings.size(-1),
                        coverage=self.coverage,
                        masking=self.masking,
                        device=self.device,
                        generator=g,
                    )
                    rep_a = self.aggregator(modality_embeddings * mask_a)
                    rep_b = self.aggregator(modality_embeddings * mask_b)
                    loss = vicreg_loss(
                        rep_a,
                        rep_b,
                        sim_coeff=self.sim_coeff,
                        std_coeff=self.std_coeff,
                        cov_coeff=self.cov_coeff,
                        std_const=self.std_const,
                    )
                else:
                    raise ValueError(f"Unsupported pretraining_type: {self.pretraining_type}")

                for optimizer in optimizers:
                    optimizer.zero_grad()
                if aggregator_optimizer is not None:
                    aggregator_optimizer.zero_grad()
                if self.scale_optimizer is not None:
                    self.scale_optimizer.zero_grad()
                loss.backward()
                for optimizer in optimizers:
                    optimizer.step()
                if aggregator_optimizer is not None:
                    aggregator_optimizer.step()
                if self.scale_optimizer is not None:
                    self.scale_optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch}/{self.epochs}, Loss: {avg_loss:.4f}")
        
    def save_encoders(self):
        pretraining_folder = config_folder + '/../multimodal_enc/pretraining/'
        for i, encoder in enumerate(self.encoders):
            save_path = os.path.join(pretraining_folder, f'encoder_{self.chosen_label}_{self.data_type}_{self.pretraining_type}_{self.pretrain_model}{self.conv_layers}_batch{self.batch_size}_epoch{self.epochs}_feat_{self.feature_tag}_{i}_pretrained.pt')
            torch.save({
                'encoder_state_dict': encoder.state_dict(),
                'logit_scale': self.logit_scale.item(),
                'encoder_config': {
                    'input_features': self.feature_split[i],
                    'conv_layers': self.conv_layers,
                    'dropout': self.dropout,
                    'num_heads': 4 if self.pretrain_model.upper() == 'MHA' else 0
                }
            }, save_path)
            print(f"Saved encoder {i} to {save_path}")

        if self.pretraining_type == 'regularization-vicreg' and self.aggregator is not None:
            self.save_aggregator(pretraining_folder)

    def save_aggregator(self, pretraining_folder=None):
        pretraining_folder = config_folder + '/../multimodal_enc/pretraining/'
        save_path = os.path.join(
            pretraining_folder,
            f'aggregator_{self.chosen_label}_{self.data_type}_{self.pretraining_type}_{self.pretrain_model}{self.conv_layers}_batch{self.batch_size}_epoch{self.epochs}_feat_{self.feature_tag}_pretrained.pt'
        )
        torch.save({
            'aggregator_state_dict': self.aggregator.state_dict(),
            'aggregator_config': {
                'num_feature_groups': len(self.feature_split),
                'embedding_dim': self.embedding_dim,
            }
        }, save_path)
        print(f"Saved aggregator to {save_path}")
    
    def get_encoders(self):
        return self.encoders
    
    def get_aggregator(self):
        return self.aggregator
