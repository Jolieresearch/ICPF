
    
    
import torch
from torch import nn

from models.angle import AnglEModel_ICPF
from models.vit import VitModel_ICPF
from utils.module import AttentionFusion


class LabelEmbedding(nn.Module):
    def __init__(self, num_retrieved, device, hidden_dim=768):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(num_retrieved, hidden_dim)).to(device)
    
    def forward(self, x):
        x = x.unsqueeze(-1)
        embedded = x * self.embedding
        return self.layer_norm(embedded)
    

class ICPF(torch.nn.Module):
    def __init__(self, angle_model_path, vit_model_path, prompt_re_length, retrieved_num, device):
        super(ICPF, self).__init__()
        self.t_feat_dim = 768
        self.v_feat_dim = 768
        self.hidden_dim = 768
        self.device = device
        self.retrieved_num = retrieved_num

        self.angle_model = AnglEModel_ICPF(angle_model_path, prompt_re_length, retrieved_num, device)
        self.vit_model = VitModel_ICPF(vit_model_path, prompt_re_length, retrieved_num, device)

        self.text_proj = nn.Linear(self.t_feat_dim, self.hidden_dim)
        self.image_proj = nn.Linear(self.v_feat_dim, self.hidden_dim)
        self.label_embedding = LabelEmbedding(self.retrieved_num, self.hidden_dim, self.device)
        
        self.t_dnns = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.v_dnns = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.IMT = AttentionFusion()
        self.final_linear = nn.Linear(2 * self.hidden_dim, 1)

    def forward(self, text, retrieved_text_list_text, transform_video_data, retrieved_visual_feature_embedding_video, retrieved_label_video, retrieved_label_text):
        """
        Forward pass of the ICPF model.

        Args:
            text (torch.Tensor): Input text tensor.
            retrieved_text_list_text (List[str]): List of retrieved texts.
            transform_video_data (torch.Tensor): Input video tensor.
            retrieved_visual_feature_embedding_video (torch.Tensor): Retrieved visual feature embeddings.
            retrieved_label_video (torch.Tensor): Retrieved video labels.
            retrieved_label_text (torch.Tensor): Retrieved text labels.

        Returns:
            torch.Tensor: Output tensor.
        """

        t_feat = self.angle_model.encode(text, retrieved_text_list_text)
        v_feat = self.vit_model(transform_video_data, retrieved_visual_feature_embedding_video)

        label_video = retrieved_label_video
        label_text = retrieved_label_text
        
        t_embs = self.text_proj(t_feat)
        v_embs = self.image_proj(v_feat)
        text_label_embs = self.label_embedding(label_text)
        video_label_embs = self.label_embedding(label_video)
        label_embs = torch.cat([text_label_embs, video_label_embs], dim=1)
        label_embs = torch.mean(label_embs, dim=1, keepdim=True)
        
        t_embs = t_embs.to(self.device)
        v_embs = v_embs.to(self.device)
        label_embs = label_embs.to(self.device)

        text_length = t_feat.size(1) // 2
        text_mask = torch.narrow(t_feat, 1, text_length, text_length).to(self.device)
        image_mask = torch.ones((t_feat.size(0), 20), device=self.device)

        t_embs, v_embs = self.IMT(label_embs, t_embs, text_mask, v_embs, image_mask)

        t_embs = self.t_dnns(t_embs)
        v_embs = self.v_dnns(v_embs)

        output = torch.cat((t_embs, v_embs), dim=1)
        output = self.final_linear(output)

        return output

