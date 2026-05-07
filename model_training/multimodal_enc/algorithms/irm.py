import torch
import torch.nn as nn
from .base import BaseDGAlgorithm

class IRM(BaseDGAlgorithm):
    def __init__(self, model, device, penalty_weight=1e4):
        super().__init__(model, device)
        self.penalty_weight = penalty_weight

    def __call__(self, *train_batch, criterion=None):
        criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        # train_batch expected: (x1, x2, x3, y, group_labels)
        x1_batch, x2_batch, x3_batch, y_batch, groups = train_batch
        _, logits = self.model(x1_batch, x2_batch, x3_batch)
        # ERM component
        loss_erm = criterion(logits, y_batch)
        # IRMv1 penalty: gradient of risk w.r.t. a dummy scalar should be ~0 per environment
        unique_users = torch.unique(groups)
        penalty_terms = []
        for u in unique_users:
            idx = (groups == u).nonzero(as_tuple=True)[0]
            if idx.numel() < 2:
                continue
            logits_u = logits.index_select(0, idx)
            y_u = y_batch.index_select(0, idx)
            scale = torch.tensor(1.0, device=self.device, requires_grad=True)
            loss_scaled = criterion(scale * logits_u, y_u)
            g = torch.autograd.grad(loss_scaled, [scale], create_graph=True)[0]
            # penalty is grad^2 (not the loss itself)
            penalty_terms.append(g.pow(2))
        irm_penalty = (
            torch.stack(penalty_terms).mean() if len(penalty_terms) > 0
            else torch.tensor(0.0, device=self.device)
        )
        loss = loss_erm + self.penalty_weight * irm_penalty
        # Rescale the entire loss to keep gradients in a reasonable range
        # loss /= self.penalty_weight
        # loss = irm_penalty
        return loss, irm_penalty * self.penalty_weight