import math

import torch
from torch import nn


class PositionalEmbedding(nn.Module):
    """
    Sinusoidal positional embeddings. This module expects the text embeddings as input as (B, T, E) and already adds the
    positional embedding to this tensor.
    """

    def __init__(self, model_dim, context_len, dropout, device):
        super(PositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        positional_embedding = torch.zeros(context_len, model_dim, device=device)
        position = torch.arange(
            0, context_len, dtype=torch.float, device=device
        ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, model_dim, 2, device=device).float()
            * (-math.log(10000.0) / model_dim)
        )
        positional_embedding[:, 0::2] = torch.sin(position * div_term)
        positional_embedding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("positional_embedding", positional_embedding)

    def forward(self, context):
        pos_embedding = self.positional_embedding.repeat(context.size(0), 1, 1)
        context = context + pos_embedding[:, : context.size(1), :]
        return self.dropout(context)
