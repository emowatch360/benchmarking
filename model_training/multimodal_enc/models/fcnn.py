import torch
import torch.nn as nn

class FCNN_Encoder(nn.Module):
    def __init__(self, input_dim, fcnn_layers, dropout):
        super(FCNN_Encoder, self).__init__()
        layers = []
        assert len(fcnn_layers) >= 1, "fcnn_layers must have at least one layer size specified."
        # first layer
        layers.append(nn.Linear(input_dim, fcnn_layers[0]))
        layers.append(nn.LayerNorm(fcnn_layers[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        # hidden layers
        for layer_num in range(1, len(fcnn_layers)):
            layers.append(nn.Linear(fcnn_layers[layer_num-1], fcnn_layers[layer_num]))
            layers.append(nn.LayerNorm(fcnn_layers[layer_num]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
