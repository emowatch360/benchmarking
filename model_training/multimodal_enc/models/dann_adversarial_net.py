# model.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .multimodal_encoder import MultimodalEncoderNet

class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer from Algorithm 1 (lines 17-31 in the paper)
    Forward pass: identity function
    Backward pass: gradient is multiplied by -lambda
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod # this method does not require self/cls as input, it's more like a utility function
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambda_
        return output, None

# nn.module maintains state while torch.autograd.Function is stateless
class GradientReversalLayer(nn.Module):
    """
    create a gradient reversal layer which applies the GradientReversalFunction
    """
    def __init__(self):
        super(GradientReversalLayer, self).__init__()
    
    def forward(self, x, lambda_=1.0):
        return GradientReversalFunction.apply(x, lambda_)

class DomainAdversarialNet(MultimodalEncoderNet):
    def __init__(self, physio_input_dim, sleep_input_dim, demo_input_dim, proj_head_dim,
                 conv_layers, sp_conv_layers, fcnn_layers, dropout, num_domains, output_dim=2, feature_split=[], model_type='cnn',
                 pretrained_cp_encoder=[], pretrained_sp_encoder=[], pretrained_cp_aggregator=None, pretrained_sp_aggregator=None):
        super().__init__(physio_input_dim, sleep_input_dim, demo_input_dim, proj_head_dim, conv_layers, sp_conv_layers, fcnn_layers,
                         dropout, output_dim=output_dim, feature_split=feature_split, model_type=model_type,
                         pretrained_cp_encoder=pretrained_cp_encoder, pretrained_sp_encoder=pretrained_sp_encoder,
                         pretrained_cp_aggregator=pretrained_cp_aggregator, pretrained_sp_aggregator=pretrained_sp_aggregator)
        self.num_domains = num_domains
        if self.num_domains < 2:
            raise ValueError("num_domains must be at least 2 for domain adversarial training.")
        # Domain classification layer; models probability that a given input is from source domain or target domain
        self.gradient_reversal = GradientReversalLayer() # identity layer in the forward pass
        # Domain classification layer; predicts domain (Original paper used logistic regression)
        self.hidden_dim = self.proj_head_dim
        self.domain_classifier = nn.Linear(self.hidden_dim, self.num_domains)

    def forward(self, physio_data, sleep_data, demo_data, lambda_=1.0):
        processed_features, label_logits = super().forward(physio_data, sleep_data, demo_data)
        # Apply domain classification
        reversed_features = self.gradient_reversal(processed_features, lambda_)
        domain_logits = self.domain_classifier(reversed_features)
        return processed_features, label_logits, domain_logits
