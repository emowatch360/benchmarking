# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .fcnn import FCNN_Encoder
from .conv1d import Conv1D_Encoder
from .resnet1d import ResNet1D_Encoder
from .mha1d import MultiHeadAttention1DEncoder
from ..pretraining.crossl_utils import CroSSLAggregator

class MultimodalEncoderNet(nn.Module):
    def __init__(self, physio_input_dim, sleep_input_dim, demo_input_dim, proj_head_dim,
                 conv_layers, sp_conv_layers, fcnn_layers, dropout, output_dim=2, num_heads=4, feature_split=[], model_type='cnn',
                 pretrained_cp_encoder=[], pretrained_sp_encoder=[], pretrained_cp_aggregator=None, pretrained_sp_aggregator=None): # 'cnn' or 'resnet'
        super(MultimodalEncoderNet, self).__init__()
       
        self.physio_input_dim = physio_input_dim
        self.sleep_input_dim = sleep_input_dim
        self.demo_input_dim = demo_input_dim
        self.proj_head_dim = proj_head_dim
        self.feature_split = feature_split

        self.physio_models = nn.ModuleList([])
        self.sleep_models = nn.ModuleList([])
        self.pretrained_cp_encoder = nn.ModuleList(pretrained_cp_encoder)
        self.pretrained_sp_encoder = nn.ModuleList(pretrained_sp_encoder)
        self.physio_aggregator = None
        self.sleep_aggregator = None
        self.pretrained_cp_aggregator = pretrained_cp_aggregator
        self.pretrained_sp_aggregator = pretrained_sp_aggregator

        self.final_input_dim = 0

        # CNN for physiological features
        if self.physio_input_dim > 0:
            assert sum(self.feature_split) == physio_input_dim, 'Feature split must match physio input dimensions!'
            self.final_input_dim = self.final_input_dim + conv_layers[-1]
            if len(self.pretrained_cp_encoder) > 0:
                # If the pretrained_cp_encoder is None in either this class, in train.py, or in main.py, then this will not be triggered, skipping the SSL pretrained encoder
                self.physio_models = self.pretrained_cp_encoder
            else:
                for i in range(0, len(self.feature_split)):
                    if model_type.upper() == 'CNN':
                        self.physio_models.append(Conv1D_Encoder(input_dim=self.feature_split[i], conv_layers=conv_layers, dropout=dropout))
                    elif model_type.upper() == 'RESNET':
                        # only the last conv layer's output channels are used as base_channels !!
                        assert len(conv_layers) == 1, "ResNet1D_Encoder only supports a single conv layer configuration (the last one) for determining the base channels."
                        self.physio_models.append(ResNet1D_Encoder(input_dim=self.feature_split[i], base_channels=conv_layers[-1], dropout=dropout))
                    elif model_type.upper() == 'MHA':
                        # only the last conv layer's output channels are used as embed_dim !!
                        assert len(conv_layers) == 1, "MultiHeadAttention1DEncoder only supports a single conv layer configuration (the last one) for determining the embedding dimension."
                        self.physio_models.append(MultiHeadAttention1DEncoder(input_dim=self.feature_split[i], embed_dim=conv_layers[-1], num_heads=num_heads, dropout=dropout))
            if self.pretrained_cp_aggregator is not None:
                self.physio_aggregator = self.pretrained_cp_aggregator
            else:
                self.physio_aggregator = CroSSLAggregator(
                    num_feature_groups=len(feature_split),
                    embedding_dim=conv_layers[-1]
                )

        # CNN for sleep features
        if self.sleep_input_dim > 0:
            assert sum(self.feature_split) == sleep_input_dim, 'Feature split must match sleep input dimensions!'
            self.final_input_dim = self.final_input_dim + sp_conv_layers[-1]
            if len(self.pretrained_sp_encoder) > 0:
                # If the pretrained_sp_encoder is None in either this class, in train.py, or in main.py, then this will not be triggered, skipping the SSL pretrained encoder
                self.sleep_models = self.pretrained_sp_encoder
            else:
                for i in range(0, len(self.feature_split)):
                    if model_type.upper() == 'CNN':
                        self.sleep_models.append(Conv1D_Encoder(input_dim=self.feature_split[i], conv_layers=sp_conv_layers, dropout=dropout))
                    elif model_type.upper() == 'RESNET':
                        # only the last conv layer's output channels are used as base_channels !!
                        assert len(sp_conv_layers) == 1, "ResNet1D_Encoder only supports a single conv layer configuration (the last one) for determining the base channels."
                        self.sleep_models.append(ResNet1D_Encoder(input_dim=self.feature_split[i], base_channels=sp_conv_layers[-1], dropout=dropout))
                    elif model_type.upper() == 'MHA':
                        # only the last conv layer's output channels are used as embed_dim !!
                        assert len(sp_conv_layers) == 1, "MultiHeadAttention1DEncoder only supports a single conv layer configuration (the last one) for determining the embedding dimension."
                        self.sleep_models.append(MultiHeadAttention1DEncoder(input_dim=self.feature_split[i], embed_dim=sp_conv_layers[-1], num_heads=num_heads, dropout=dropout))
            if self.pretrained_sp_aggregator is not None:
                self.sleep_aggregator = self.pretrained_sp_aggregator
            else:
                self.sleep_aggregator = CroSSLAggregator(
                    num_feature_groups=len(feature_split),
                    embedding_dim=sp_conv_layers[-1]
                )
        
        # MLP for demographic features
        if self.demo_input_dim > 0:
            self.final_input_dim = self.final_input_dim + fcnn_layers[-1]
            self.demo_mlp = FCNN_Encoder(input_dim=demo_input_dim, fcnn_layers=fcnn_layers, dropout=dropout)

        assert self.final_input_dim > 0, "At least one input modality must be provided."
        
        self.projection_head = nn.Sequential(
            nn.Linear(self.final_input_dim, self.proj_head_dim),  # 128 features from each of the networks
            nn.LayerNorm(self.proj_head_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(self.proj_head_dim, output_dim)
    
    def forward(self, physio_data, sleep_data, demo_data):
        
        # Process each modality
        final_feature_list = []
        
        if self.physio_input_dim > 0:
            split_physio_data = torch.split(physio_data, self.feature_split, dim=2)
            
            # Average pooling encoder outputs before going through the projection head and classifier
            if len(self.pretrained_cp_encoder) > 0:
                encoder_outputs = torch.stack(
                    [self.pretrained_cp_encoder[i](split_physio_data[i]) for i in range(len(self.feature_split))],
                    dim=1  # shape: (batch, num_splits, hidden_dim)
                )
                final_feature_list.append(self.physio_aggregator(encoder_outputs))
            else: 
                physio_outputs = torch.stack(
                    [self.physio_models[i](split_physio_data[i]) for i in range(len(self.feature_split))],
                    dim=1
                )
                final_feature_list.append(self.physio_aggregator(physio_outputs))
        
        if self.sleep_input_dim > 0:
            split_sleep_data = torch.split(sleep_data, self.feature_split, dim=2)
            
            # Average pooling encoder outputs before going through the projection head and classifier
            if len(self.pretrained_sp_encoder) > 0:
                encoder_outputs = torch.stack(
                    [self.pretrained_sp_encoder[i](split_sleep_data[i]) for i in range(len(self.feature_split))],
                    dim=1  # shape: (batch, num_splits, hidden_dim)
                )
                final_feature_list.append(self.sleep_aggregator(encoder_outputs))
            else: 
                sleep_outputs = torch.stack(
                    [self.sleep_models[i](split_sleep_data[i]) for i in range(len(self.feature_split))],
                    dim=1
                )
                final_feature_list.append(self.sleep_aggregator(sleep_outputs))
        
        if self.demo_input_dim > 0:
            vector_output = self.demo_mlp(demo_data)
            final_feature_list.append(vector_output)
        
        # Concatenate all features
        aggregated_features = torch.cat(final_feature_list, dim=1)
        processed_features = self.projection_head(aggregated_features)
        
        # Final classification
        output = self.classifier(processed_features)
        return processed_features, output
