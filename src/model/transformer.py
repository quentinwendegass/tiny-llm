from torch import nn

from model.multi_headed_masked_attention import MultiHeadedMaskedAttention


class Transformer(nn.Module):
    """
    Decoder only transformer with masked self-attention. In contrast to the transformer in the original paper,
    the normalization layers are applied before the attention and feed-forward layers.
    """

    def __init__(self, model_dim, num_heads, num_hidden, device, dropout=0.5):
        super(Transformer, self).__init__()
        self.device = device
        self.multi_head_attention = MultiHeadedMaskedAttention(
            model_dim, num_heads, dropout, device
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, num_hidden),
            nn.GELU(),
            nn.Linear(num_hidden, model_dim),
        )
        self.multi_head_attention_normalization = nn.LayerNorm(model_dim)
        self.feed_forward_normalization = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, context):
        # Masked multi head attention
        context = self.multi_head_attention_normalization(context)
        attention_output = self.multi_head_attention(context)
        context = context + self.dropout(attention_output)

        # Feed forward
        context = self.feed_forward_normalization(context)
        feed_forward_output = self.feed_forward(context)
        context = context + self.dropout(feed_forward_output)

        return context
