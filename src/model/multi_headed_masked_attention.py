import torch
from torch import nn


class MultiHeadedMaskedAttention(nn.Module):
    class SingleHeadAttention(nn.Module):
        def __init__(self, model_dim, head_size, device):
            super().__init__()
            self.device = device
            self.key_layer = nn.Linear(model_dim, head_size, bias=False)
            self.query_layer = nn.Linear(model_dim, head_size, bias=False)
            self.value_layer = nn.Linear(model_dim, head_size, bias=False)

        def forward(self, embedded):
            k = self.key_layer(embedded)
            q = self.query_layer(embedded)
            v = self.value_layer(embedded)

            scores = q @ torch.transpose(k, 1, 2)
            context_length, attention_dim = k.shape[1], k.shape[2]
            scores = scores / (attention_dim**0.5)

            lower_triangular = torch.tril(torch.ones(context_length, context_length))
            mask = (lower_triangular == 0).to(self.device)
            scores = scores.masked_fill(mask, float("-inf"))
            scores = nn.functional.softmax(scores, dim=2)

            return scores @ v

    def __init__(self, model_dim, num_heads, dropout, device):
        super().__init__()
        self.attention_heads = nn.ModuleList()
        for i in range(num_heads):
            self.attention_heads.append(
                self.SingleHeadAttention(model_dim, model_dim // num_heads, device)
            )
        self.compute = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, embedded):
        head_outputs = []
        for head in self.attention_heads:
            head_outputs.append(head(embedded))
        concatenated = torch.cat(head_outputs, dim=2)
        return self.dropout(self.compute(concatenated))
