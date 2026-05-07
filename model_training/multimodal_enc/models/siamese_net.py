# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .multimodal_encoder import MultimodalEncoderNet

class SiameseNet(MultimodalEncoderNet):
    """Siamese network for 1D time-series inputs.
    forward_once(x): returns an L2-normalized embedding vector.
    """

    def __init__(self, physio_input_dim, sleep_input_dim, demo_input_dim,
                 proj_head_dim, conv_layers, sp_conv_layers, fcnn_layers,
                 dropout, output_dim=2, num_heads=4, feature_split=[],
                 model_type='cnn'):
        super().__init__(
            physio_input_dim=physio_input_dim,
            sleep_input_dim=sleep_input_dim,
            demo_input_dim=demo_input_dim,
            proj_head_dim=proj_head_dim,
            conv_layers=conv_layers,
            sp_conv_layers=sp_conv_layers,
            fcnn_layers=fcnn_layers,
            dropout=dropout,
            output_dim=output_dim,
            num_heads=num_heads,
            feature_split=feature_split,
            model_type=model_type,
        )
        self.embedding_dim = proj_head_dim

    def forward_once(self, physio_data, sleep_data, demo_data) -> torch.Tensor:
        """Compute embedding for a single branch.
        Returns:
            Tensor of shape (batch_size, embedding_dim), L2-normalized.
        """
        embedding_mat, _ = super().forward(physio_data, sleep_data, demo_data)
        assert embedding_mat.dim() == 2 and embedding_mat.size(1) == self.embedding_dim, "Incorrect embedding dim!"
        embedding_mat = F.normalize(embedding_mat, p=2, dim=1) # L2-normalize embeddings
        return embedding_mat
