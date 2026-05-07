import torch
import torch.nn as nn
from .base import BaseDGAlgorithm

class VREX(BaseDGAlgorithm):
    def __init__(self, model, device, vrex_lambda):
        # domains returns all the unique group names
        super().__init__(model, device)
        self.vrex_lambda = vrex_lambda

    # implement VRex algorithm - variance risk extrapolation
    def __call__(self, *train_batch, criterion=None):
        criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        # train_batch expected: (x1, x2, x3, y, group_labels)
        x1_batch, x2_batch, x3_batch, y_batch, groups = train_batch
        _, logits = self.model(x1_batch, x2_batch, x3_batch)
        unique_users = torch.unique(groups)
        domain_loss_list = []
        for u in unique_users:
            idx = (groups == u).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            logits_u = logits.index_select(0, idx)
            y_u = y_batch.index_select(0, idx)
            # compute loss per user
            loss_per_domain = criterion(logits_u, y_u) # default reduction is 'mean'
            domain_loss_list.append(loss_per_domain)
        if len(domain_loss_list) <= 1:
            # if less than 2 domains in the batch, return ERM loss
            return criterion(logits, y_batch), torch.tensor(0.0, device=self.device)
        domain_loss_list = torch.stack(domain_loss_list)
        mean_loss = domain_loss_list.mean()
        variance_loss = ((domain_loss_list - mean_loss.detach()) ** 2).mean()
        # updating the variance calculation to not use torch.var to ensure detach works correctly
        # variance_loss =  domain_loss_list.var(unbiased=False) # unbiased=False for population variance (i.e., divide by N, not N-1)
        loss = mean_loss + variance_loss * self.vrex_lambda
        # loss /= self.vrex_lambda # added this to scale down the gradients
        return loss, variance_loss * self.vrex_lambda