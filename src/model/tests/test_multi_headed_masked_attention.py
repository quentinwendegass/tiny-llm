import unittest

import torch
from torch import nn

from src.model.multi_headed_masked_attention import MultiHeadedMaskedAttention


class MultiHeadAttentionTest(unittest.TestCase):
    def test_torch_and_custom_behave_the_same(self):
        torch.manual_seed(200)

        embed_dim = 2
        num_heads = 2
        torch_mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        custom_mha = MultiHeadedMaskedAttention(
            embed_dim, num_heads, 0, device=torch.device("cpu")
        )

        model_input = torch.tensor(
            [[[0, 1], [1, 1], [4, 1]], [[2, 1], [5, 1], [4, 1]]], dtype=torch.float
        )

        with torch.no_grad():
            # set the same weights
            torch_mha.in_proj_weight.data.fill_(0.5)
            torch_mha.out_proj.weight.data.fill_(0.5)
            custom_mha.attention_heads[0].key_layer.weight.data.fill_(0.5)
            custom_mha.attention_heads[0].value_layer.weight.data.fill_(0.5)
            custom_mha.attention_heads[0].query_layer.weight.data.fill_(0.5)
            custom_mha.attention_heads[1].key_layer.weight.data.fill_(0.5)
            custom_mha.attention_heads[1].value_layer.weight.data.fill_(0.5)
            custom_mha.attention_heads[1].query_layer.weight.data.fill_(0.5)
            custom_mha.compute.weight.data.fill_(0.5)
            custom_mha.compute.bias.data.fill_(0)

            o = torch_mha(
                model_input,
                model_input,
                model_input,
                attn_mask=torch.nn.Transformer.generate_square_subsequent_mask(
                    model_input.size(1)
                ),
            )
            o2 = custom_mha(model_input)

            self.assertEqual(o[0].tolist(), o2.tolist())
