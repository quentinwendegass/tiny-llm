import math

import torch
from torch import nn

from model.positional_embedding import PositionalEmbedding
from model.transformer import Transformer


class GPT(nn.Module):
    def __init__(
        self,
        num_token,
        model_dim,
        num_head,
        num_hidden,
        num_blocks,
        context_len,
        dropout=0.1,
        pretrained_embeddings=None,
        device=torch.device("cpu"),
    ):
        super(GPT, self).__init__()
        self.device = device
        if pretrained_embeddings is not None:
            self.text_embedding = nn.Embedding.from_pretrained(
                torch.tensor(pretrained_embeddings, dtype=torch.float32).to(device),
                freeze=False,
            )
            self.model_dim = pretrained_embeddings.shape[1]
        else:
            self.text_embedding = nn.Embedding(num_token, model_dim).to(device)
            self.model_dim = model_dim

        self.positional_embedding = PositionalEmbedding(
            self.model_dim, context_len, dropout, device
        )

        self.transformers = nn.ModuleList(
            [
                Transformer(self.model_dim, num_head, num_hidden, device, dropout)
                for _ in range(num_blocks)
            ]
        )

        self.normalization = nn.LayerNorm(self.model_dim)
        self.decoder = nn.Linear(self.model_dim, num_token)

    def forward(self, context):
        context = self.text_embedding(context) * math.sqrt(self.model_dim)
        context = self.positional_embedding(context)

        for transformer in self.transformers:
            context = transformer(context)

        context = self.normalization(context)
        return self.decoder(context)
