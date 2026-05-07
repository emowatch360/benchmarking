import torch
import torch.nn as nn

class BaseDGAlgorithm:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.to(self.device)

    def __call__(self, *train_batch, criterion=None):
        # implement ERM
        criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        # train_batch expected: (x1, x2, x3, y, group_labels)
        x1_batch, x2_batch, x3_batch, y_batch, _ = train_batch
        _, logits = self.model(x1_batch, x2_batch, x3_batch)
        loss = criterion(logits, y_batch)
        return loss, torch.Tensor([0.0])  # returning 0.0 as a placeholder for any additional metric