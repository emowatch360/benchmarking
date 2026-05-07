import torch
import torch.nn as nn
from .base import BaseDGAlgorithm
from model_training.utils.common_utils import convert_categrical_to_zero_indexed

class DANN(BaseDGAlgorithm):
    def __init__(self, model, device, dann_lambda, domains):
        # domains returns all the unique group names
        super().__init__(model, device)
        self.dann_lambda = dann_lambda
        self.domains = domains

    # implement DANN algorithm - variance risk extrapolation
    def __call__(self, *train_batch, criterion=None):
        criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        # train_batch expected: (x1, x2, x3, y, group_labels)
        x1_batch, x2_batch, x3_batch, y_batch, groups = train_batch
        _, label_logits, domain_logits = self.model(x1_batch, x2_batch, x3_batch, lambda_=self.dann_lambda)
        # logits loss
        logits_loss = criterion(label_logits, y_batch)
        # domain loss
        mapped_groups = convert_categrical_to_zero_indexed(groups, all_groups=self.domains)
        if mapped_groups.max().item() >= domain_logits.size(1):
            raise ValueError(
                f"Domain target index {mapped_groups.max().item()} exceeds "
                f"domain classifier output size {domain_logits.size(1)}."
            )
        domain_loss = criterion(domain_logits, mapped_groups) # ensure that groups are in [0, num_domains-1]
        loss = logits_loss + domain_loss
        return loss, domain_loss
