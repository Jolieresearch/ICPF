import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_inner, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_inner)
        self.w_2 = nn.Linear(d_inner, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.activate = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.dropout(self.w_2(self.activate(self.w_1(x))))
        return self.layer_norm(residual + x)

class SelfAttention(nn.Module):
    def __init__(self, temperature, dropout):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value, mask):
        attn = torch.matmul(query, key.transpose(-2, -1)) / self.temperature
        attn = attn + mask
        p_attn = self.dropout(self.softmax(attn))
        return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads, d_model, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.d_v = self.d_k

        self.w_Q = nn.Linear(d_model, n_heads * self.d_k, bias=False)
        self.w_K = nn.Linear(d_model, n_heads * self.d_k, bias=False)
        self.w_V = nn.Linear(d_model, n_heads * self.d_v, bias=False)
        self.fc = nn.Linear(n_heads * self.d_v, d_model, bias=False)

        self.self_attention = SelfAttention(
            temperature=self.d_k**0.5, dropout=dropout
        )
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, query, key, value, mask):
        sz_b, len_q, len_k, len_v = (
            query.size(0),
            query.size(1),
            key.size(1),
            value.size(1),
        )
        residual = query

        q = self.w_Q(query).view(sz_b, len_q, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_K(key).view(sz_b, len_k, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_V(value).view(sz_b, len_v, self.n_heads, self.d_v).transpose(1, 2)

        x, attn = self.self_attention(q, k, v, mask=mask)
        x = x.transpose(1, 2).contiguous().view(sz_b, len_q, self.d_model)
        x = self.dropout(self.fc(x))
        return self.layer_norm(residual + x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_inner, dropout):
        super().__init__()
        self.multi_head_attention = MultiHeadedAttention(
            n_heads=n_heads, d_model=d_model, dropout=dropout
        )
        self.feed_forward = PositionwiseFeedForward(
            d_model=d_model, d_inner=d_inner, dropout=dropout
        )

    def forward(self, block_input, mask):
        output = self.multi_head_attention(block_input, block_input, block_input, mask)
        return self.feed_forward(output)


class AttentionFusion(torch.nn.Module):
    def __init__(self):
        super(AttentionFusion, self).__init__()
        d_model = 768
        dropout = 0.1
        n_heads = 1
        n_layers = 1
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.token_type_embeddings = nn.Embedding(2, d_model)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_inner=d_model * 4,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        label_feats,
        lang_feats,
        lang_attention_mask,
        visn_feats,
        visn_attention_mask,
    ):
        text_length = lang_feats.size(1)

        lang_att_output = lang_feats
        visn_att_output = visn_feats
        label_att_output = label_feats

        lang_attention_mask = lang_attention_mask.unsqueeze(1)
        lang_attention_mask = lang_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        lang_attention_mask = (1.0 - lang_attention_mask) * -10000.0
        lang_attention_mask = torch.mean(lang_attention_mask, dim=2, keepdim=True)

        visn_attention_mask = visn_attention_mask.unsqueeze(1).unsqueeze(2)
        visn_attention_mask = visn_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        visn_attention_mask = (1.0 - visn_attention_mask) * -10000.0

        visn_att_output = torch.mean(visn_att_output, dim=1)
        
        att_output = torch.cat([lang_att_output, visn_att_output, label_att_output], dim=1)
        att_output = self.dropout(att_output)
        mask_output = torch.cat([lang_attention_mask, visn_attention_mask], dim=-1)
        mask_output = torch.narrow(mask_output, 3, 0, lang_att_output.size(1) + visn_att_output.size(1) + label_att_output.size(1))
        mask_output = mask_output.expand(-1, -1, lang_att_output.size(1) + visn_att_output.size(1) + label_att_output.size(1), lang_att_output.size(1) + visn_att_output.size(1) + label_att_output.size(1))

        for transformer in self.transformer_blocks:
            att_output = transformer.forward(att_output, mask_output)

        # return att_outputs
        x = att_output[:, :text_length, :][:, 0]
        y = att_output[:, text_length:, :][:, 0]
        return x, y
