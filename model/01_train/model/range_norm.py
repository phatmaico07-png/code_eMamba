"""eMamba Range Normalization (replaces LayerNorm).

y = gamma * (x - mean) / range(x - mean) + beta,  range = max - min

Uses only mean + max/min + divide (no sqrt / variance) -> hardware-friendly and
quantization-robust. Root idea: Range Batch-Norm (Banner et al., NeurIPS 2018).
Normalizes over the last dimension (per token, across features).
"""
import torch
import torch.nn as nn


class RangeNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):  # x: (..., dim)
        xc = x - x.mean(dim=-1, keepdim=True)
        rng = xc.max(dim=-1, keepdim=True).values - xc.min(dim=-1, keepdim=True).values
        return self.gamma * (xc / (rng + self.eps)) + self.beta
