import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention1DEncoder(nn.Module):
    """
    Multi-Head Attention (MHA) 1D encoder.
        input_proj:     Linear(input_dim -> embed_dim)
        qkv_proj:       Linear(embed_dim -> 3 * embed_dim)  (combined Q,K,V)
        split_heads:    (B, T, embed_dim) -> (B, num_heads, T, head_dim)
        attention:      scaled dot-product self-attention over time
        combine_heads:  (B, num_heads, T, head_dim) -> (B, T, embed_dim)
        attended.mean:  mean over time -> (B, embed_dim)
    Expected input:
        x: Tensor of shape (batch, time_steps, features) == (B, T, C)
    Output:
        Tensor of shape (batch, embed_dim), e.g. (B, 128)
    """

    def __init__(self, input_dim: int, embed_dim: int = 128, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        # input_proj: project input features to embedding dimension
        # (B, T, C) -> (B, T, embed_dim)
        self.input_proj = nn.Linear(input_dim, embed_dim) # [B, 30, 28] -> [B, 30, 128] => here time dimension is equivalent to a token of embedding size 128
        # qkv_proj: combined Q, K, V projection
        # (B, T, embed_dim) -> (B, T, 3 * embed_dim)
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.attn_dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split the last dimension into (num_heads, head_dim) and transpose.

        Input:  x of shape (B, T, embed_dim)
        Output: x of shape (B, num_heads, T, head_dim)
        """
        B, T, E = x.shape
        x = x.view(B, T, self.num_heads, self.head_dim)  # (B, T, H, D)
        x = x.permute(0, 2, 1, 3)                        # (B, H, T, D)
        return x

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Combine heads back to (B, T, embed_dim).

        Input:  x of shape (B, num_heads, T, head_dim)
        Output: x of shape (B, T, embed_dim)
        """
        B, H, T, D = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()  # (B, T, H, D)
        x = x.view(B, T, H * D)                 # (B, T, embed_dim)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, time_steps, features) i.e. (B, T, C)

        Returns:
            Tensor of shape (batch, embed_dim), i.e. (B, 128) if embed_dim=128
        """
        # Project input features to embedding space: (B, T, C) -> (B, T, E)
        x_proj = self.input_proj(x)  # input_proj

        # Q, K, V projections: (B, T, E) -> (B, T, 3E)
        qkv = self.qkv_proj(x_proj)
        q, k, v = qkv.chunk(3, dim=-1)  # each (B, T, E)

        # Split into heads: (B, T, E) -> (B, H, T, D)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # Scaled dot-product attention
        # q: (B, H, T, D), k: (B, H, T, D) -> attn_scores: (B, H, T, T)
        scale = 1.0 / (self.head_dim ** 0.5)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        # Attention output: (B, H, T, T) @ (B, H, T, D) -> (B, H, T, D)
        attn_output = torch.matmul(attn_weights, v)
        # Combine heads: (B, H, T, D) -> (B, T, E)
        combined = self._combine_heads(attn_output)
        # Reduce over time dimension (mean), as in "attended.mean" row
        # (B, T, E) -> (B, E)
        attended_mean = combined.mean(dim=1)
        return attended_mean  # (B, embed_dim)