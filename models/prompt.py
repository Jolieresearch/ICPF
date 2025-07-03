

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == q_dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(q_dim, inner_dim, bias=True)
        self.to_k = nn.Linear(k_dim, inner_dim, bias=True)
        self.to_v = nn.Linear(v_dim, inner_dim, bias=True)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, q_dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, q, k, v):
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out

class CrossFormer(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, num_layers=2, heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        self.cross_layers = nn.ModuleList([
            CrossAttention(q_dim, k_dim, v_dim, heads, dim_head, dropout) 
            for _ in range(num_layers)
        ])

    def forward(self, q, k, v):
        for layer in self.cross_layers:
            q = layer(q, k, v)
        return q


class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim=768, middle_dim=1024, output_dim=768, n_layers=2, dropout=0.1):
        super().__init__()
        layers = []
        for _ in range(n_layers):
            layers.append(nn.Sequential(
                nn.Linear(embed_dim, middle_dim),
                # nn.LayerNorm(middle_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
            embed_dim = middle_dim
        self.layers = nn.Sequential(*layers)
        self.fc_out = nn.Linear(middle_dim, output_dim)
        # self.layer_norm_out = nn.LayerNorm(output_dim)

    def forward(self, x, y):
        for layer in self.layers:
            x = layer(x)
            y = layer(y)
        x = self.fc_out(x)
        y = self.fc_out(y)
        # return self.layer_norm_out(x), self.layer_norm_out(y)
        return x, y

class LearnablePooling(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        attention_weights = self.attention(x)
        return torch.bmm(attention_weights.transpose(1, 2), x)

class Video_Prompt_Generator_Att(nn.Module):
    def __init__(self, prompt_re_length, dropout, device):
        super().__init__()
        self.device = device
        self.image_text_net = FeedForwardNetwork(embed_dim=768, middle_dim=1024, output_dim=768, n_layers=1, dropout=dropout)
        self.retrieval_fuse = CrossFormer(q_dim=768, k_dim=768, v_dim=768, num_layers=1, heads=8, dim_head=64, dropout=dropout).to(device)
        self.adaptive_prompt = LearnablePooling(768, prompt_re_length).to(device)

    def forward(self, target_video_frames_embeddings, retrieved_video_frames_embeddings):
        target_video_feature, retrieved_videos_feature = self.image_text_net(target_video_frames_embeddings, retrieved_video_frames_embeddings)
        fused_videos_features = self.retrieval_fuse(target_video_feature, retrieved_videos_feature, retrieved_videos_feature)
        video_prompt = torch.mean(fused_videos_features, dim=0, keepdim=True)
        video_prompt = self.adaptive_prompt(video_prompt)
        return video_prompt

class Text_Prompt_Generator_Att(nn.Module):
    def __init__(self, prompt_re_length, dropout, device):
        super().__init__()
        self.device = device
        self.prompt_re_length = prompt_re_length
        self.image_text_net = FeedForwardNetwork(embed_dim=768, middle_dim=1024, output_dim=768, n_layers=1, dropout=dropout).to(device)
        self.retrieval_fuse = CrossFormer(q_dim=768, k_dim=768, v_dim=768, num_layers=1, heads=8, dim_head=64, dropout=dropout).to(device)
        self.adaptive_prompt = LearnablePooling(768, prompt_re_length).to(device)

    def forward(self, target_text_embeddings, retrieved_texts_embeddings):
        retrieved_texts_embeddings = torch.mean(retrieved_texts_embeddings, dim=1).squeeze(1)
        target_text_feature, retrieved_texts_feature = self.image_text_net(target_text_embeddings, retrieved_texts_embeddings)
        fused_texts_features = self.retrieval_fuse(target_text_feature, retrieved_texts_feature, retrieved_texts_feature)
        text_prompt = self.adaptive_prompt(fused_texts_features)
        return text_prompt
