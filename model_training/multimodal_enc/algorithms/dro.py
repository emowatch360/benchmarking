import torch
import torch.nn as nn
from .base import BaseDGAlgorithm

class DRO(BaseDGAlgorithm):
    def __init__(self, model, device, eta, domains):
        # domains returns all the unique group names
        super().__init__(model, device)
        self.eta = eta
        self.domains = domains
        self.num_domains = len(domains)
        self.domain_to_idx_map = {domain: idx for idx, domain in enumerate(self.domains)}
        self.domain_weights = torch.ones(self.num_domains).to(device) / max(1, self.num_domains)

    # implement Domain Robust Optimization (DRO)
    # DRO does not have the ERM component
    def __call__(self, *train_batch, criterion=None):
        criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        # train_batch expected: (x1, x2, x3, y, group_labels)
        x1_batch, x2_batch, x3_batch, y_batch, groups = train_batch
        _, logits = self.model(x1_batch, x2_batch, x3_batch)
        unique_users = torch.unique(groups)
        domain_loss_list, batch_domain_idxes = [], []
        for u in unique_users:
            idx = (groups == u).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            logits_u = logits.index_select(0, idx)
            y_u = y_batch.index_select(0, idx)
            # compute loss per user
            loss_per_domain = criterion(logits_u, y_u)
            domain_loss_list.append(loss_per_domain)
            batch_domain_idxes.append(self.domain_to_idx_map[u.item()])
        if len(domain_loss_list) == 0:
            # if no domains in the batch, return ERM loss
            return criterion(logits, y_batch), torch.tensor(0.0, device=self.device)
        domain_loss_list = torch.stack(domain_loss_list)
        # only contains indices of the domains present in the current batch
        batch_domain_idxes = torch.tensor(batch_domain_idxes, device=self.device, dtype=torch.long)
        with torch.no_grad():
            # compute weights for each domain in the batch
            # do not normalize before multiplying -- to take the global weights into account
            batch_domain_weights = torch.exp(self.eta * domain_loss_list.detach())
            self.domain_weights[batch_domain_idxes] *= batch_domain_weights
            denom = self.domain_weights[batch_domain_idxes].sum().clamp_min(1e-12)
            self.domain_weights[batch_domain_idxes] /= denom
        loss = torch.sum(self.domain_weights[batch_domain_idxes] * domain_loss_list)
        return loss, loss
    
    def get_domain_weights(self):
        return self.domain_weights.cpu().numpy()