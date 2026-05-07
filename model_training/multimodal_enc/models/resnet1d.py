import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet1D_Encoder(nn.Module):
    """
    ResNet-style 1D encoder.
        initial_conv:  Conv1d(input_dim → base_channels, kernel=3, padding=1)
        res_block1:    2 × Conv1d + ReLU + Dropout (ResidualBlock1D)
        res_block2:    2 × Conv1d + ReLU + Dropout (ResidualBlock1D)
        pool:          MaxPool1d(kernel=2)
        flatten+proj:  Global avg pool over time → (B, base_channels)

    Expected input: x of shape (batch, time_segments, features) == (B, T, C)
    Output: feature vector of shape (batch, base_channels)
    """

    def __init__(self, input_dim: int, base_channels: list = [64], dropout: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.base_channels = base_channels
        # initial_conv: (C_in=input_dim, C_out=base_channels)
        self.initial_conv = nn.Conv1d(input_dim, base_channels, kernel_size=3, padding=1)
        self.initial_bn = nn.BatchNorm1d(base_channels)
        # two residual blocks: keep (C=base_channels, T=seq_len)
        self.res_block1 = ResidualBlock1D(base_channels, dropout=dropout)
        self.res_block2 = ResidualBlock1D(base_channels, dropout=dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling over time

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, time_segments, features) i.e. (B, T, C)

        Returns:
            Tensor of shape (batch, base_channels), i.e. (B, 64) if base_channels=64
        """
        # (B, T, C) -> (B, C, T) for Conv1d
        x = x.transpose(1, 2)  # (B, C_in, T)

        # initial_conv: (input_dim → base_channels)
        out = self.initial_conv(x)
        out = self.initial_bn(out)
        out = F.relu(out)

        # residual blocks (keep same shape: B, base_channels, T)
        out = self.res_block1(out)
        out = self.res_block2(out)
        # Global average pooling over time: (B, base_channels, T//2) -> (B, base_channels)
        out = self.pool(out) #F.adaptive_avg_pool1d(out, 1).squeeze(-1)
        out = out.squeeze(-1)  # shape: (B, base_channels)
        return out  # shape: (B, base_channels)


class ResidualBlock1D(nn.Module):
    """
    Residual block for 1D signals:
    Input / Output: (B, C, T)
    Each block: Conv1d -> BN -> ReLU -> Dropout -> Conv1d -> BN -> Dropout -> +skip -> ReLU
    """

    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.dropout1 = nn.Dropout1d(dropout)

        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout2 = nn.Dropout1d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout2(out)

        out = out + identity  # residual connection
        out = F.relu(out)
        return out


