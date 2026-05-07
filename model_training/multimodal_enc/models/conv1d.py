import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1D_Encoder(nn.Module):
    def __init__(self, input_dim, conv_layers, dropout):
        super(Conv1D_Encoder, self).__init__()

        # We use Conv1d as per https://github.com/Emognition/dl-4-tsc/blob/master/multimodal_classfiers/fcn.py
        # Conv1d uses 1d kernels as opposed to Conv2d for 2d kernels
        layers = []
        assert len(conv_layers) >= 1, "conv_layers must have at least one layer size specified."
        # first layer
        layers.append(nn.Conv1d(input_dim, conv_layers[0], kernel_size=3, padding=1))
        layers.append(nn.BatchNorm1d(conv_layers[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout1d(dropout)) # zeroes out entire channels! inplace=False by default which is good for autograd
        # hidden layers
        for layer_num in range(1, len(conv_layers)):
            layers.append(nn.Conv1d(conv_layers[layer_num-1], conv_layers[layer_num], kernel_size=3, padding=1))
            layers.append(nn.BatchNorm1d(conv_layers[layer_num]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout1d(dropout))
        layers.append(nn.AdaptiveAvgPool1d(1))  # Global average pooling over time
        self.net = nn.Sequential(*layers)
   
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, time_segments, features)
        """
        # Process input through the 2D Encoder
        x = x.transpose(1, 2)  # (batch_size, features, time_segments)
        conv_output = self.net(x)
        # Flatten the output
        conv_output = conv_output.squeeze(-1)  # (batch_size, channels)
        return conv_output
