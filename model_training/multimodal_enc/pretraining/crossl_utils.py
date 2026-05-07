import torch
import torch.nn as nn
import torch.nn.functional as F


class CroSSLAggregator(nn.Module):
    """
    CroSSL-style projector over stacked modality/feature-group embeddings.
    Mirrors the reference script:
    Flatten -> Linear(64) -> ReLU -> Linear(aggregator_dim) -> ReLU
    -> Linear(aggregator_dim) -> LayerNorm.
    """

    def __init__(self, num_feature_groups, embedding_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_feature_groups * embedding_dim, embedding_dim),
            nn.ReLU(),
            # nn.Linear(2*aggregator_dim, aggregator_dim),
            # nn.ReLU(),
            # nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, embeddings):
        return self.net(embeddings)


def get_random_masks(batch_size, num_feature_groups, embedding_dim, coverage, masking, device, generator=None):
    if masking == "random":
        mask_a = (torch.rand(batch_size, num_feature_groups, embedding_dim, device=device, generator=generator) < coverage).float()
        mask_b = (torch.rand(batch_size, num_feature_groups, embedding_dim, device=device, generator=generator) < coverage).float()
    elif masking == "spatial":
        base_mask_a = (torch.rand(batch_size, num_feature_groups, device=device, generator=generator) < coverage).float()
        base_mask_b = (torch.rand(batch_size, num_feature_groups, device=device, generator=generator) < coverage).float()
        mask_a = base_mask_a.unsqueeze(-1).expand(-1, -1, embedding_dim)
        mask_b = base_mask_b.unsqueeze(-1).expand(-1, -1, embedding_dim)
    elif masking == "temporal":
        raise ValueError("Temporal masking is not applicable because the encoder outputs are pooled feature-group embeddings.")
    else:
        raise ValueError(f"Unsupported masking strategy: {masking}")
    return mask_a, mask_b


def off_diagonal(x):
    n, m = x.shape
    if n != m:
        raise ValueError(f"Expected a square matrix, got shape {x.shape}")
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def cov_loss_each(z):
    centered = z - z.mean(dim=0, keepdim=True)
    denom = max(z.size(0) - 1, 1)
    cov = centered.T @ centered / denom
    return off_diagonal(cov).pow(2).sum() / cov.size(0)


def vicreg_loss(za, zb, sim_coeff=10.0, std_coeff=10.0, cov_coeff=100.0, std_const=1e-4):
    sim_loss = F.mse_loss(za, zb)

    za_centered = za - za.mean(dim=0, keepdim=True)
    zb_centered = zb - zb.mean(dim=0, keepdim=True)

    std_za = torch.sqrt(za_centered.var(dim=0, unbiased=False) + std_const)
    std_zb = torch.sqrt(zb_centered.var(dim=0, unbiased=False) + std_const)
    std_loss = 0.5 * (F.relu(1 - std_za).mean() + F.relu(1 - std_zb).mean())

    cov_loss = cov_loss_each(za) + cov_loss_each(zb)
    return sim_coeff * sim_loss + std_coeff * std_loss + cov_coeff * cov_loss
