import torch.nn as nn
import numpy as np
import torch
from transformers import CLIPVisionModel

from models.prompt import Video_Prompt_Generator
class VitModel_ICPF(torch.nn.Module):
    def __init__(self, vit_model_path, prompt_re_length, retrieved_num, device):

        """
        Initialize the VitModel_ICPF.

        Args:
            vit_model_path (str): Path to the pre-trained ViT model.
            prompt_re_length (int): Length of the prompt.
            retrieved_num (int): Number of retrieved instances.
            device (str): Device to run the model on ('cpu' or 'cuda').
        """

        super(VitModel_ICPF, self).__init__()
        self.device = device
        self.feature_dim = 768
        vit_model = CLIPVisionModel.from_pretrained(vit_model_path)
        self.embeddings = vit_model.vision_model.embeddings
        self.pre_layernorm = vit_model.vision_model.pre_layrnorm
        self.encoder = vit_model.vision_model.encoder
        self.encoder_layers = self.encoder.layers
        self.post_layernorm = vit_model.vision_model.post_layernorm

        vit_encoder_layer = self.encoder_layers.to(device)
        self.encoder_vit_layer0 = vit_encoder_layer[0]
        self.encoder_vit_layer1 = vit_encoder_layer[1]
        self.encoder_vit_layer2 = vit_encoder_layer[2]
        self.encoder_vit_layer3 = vit_encoder_layer[3]
        self.encoder_vit_layer4 = vit_encoder_layer[4]
        self.encoder_vit_layer5 = vit_encoder_layer[5]
        self.encoder_vit_layer6 = vit_encoder_layer[6]
        self.encoder_vit_layer7 = vit_encoder_layer[7]
        self.encoder_vit_layer8 = vit_encoder_layer[8]
        self.encoder_vit_layer9 = vit_encoder_layer[9]
        self.encoder_vit_layer10 = vit_encoder_layer[10]
        self.encoder_vit_layer11 = vit_encoder_layer[11]

        self.prompt = Video_Prompt_Generator(prompt_re_length, device=self.device)

        self.freeze(self.embeddings)
        self.freeze(self.pre_layernorm)
        self.freeze(self.encoder)
        self.freeze(self.encoder_layers)
        self.freeze(self.post_layernorm)

    def freeze(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def forward(self, transform_video_data, retrieved_visual_feature_embedding_video):
        """
        Forward pass of the model.

        Args:
            transform_video_data (List[torch.Tensor]): List of video frame tensors
            retrieved_visual_feature_embedding_video (List[torch.Tensor]): List of retrieved visual feature embeddings

        Returns:
            torch.Tensor: Processed video outputs
        """
        global layer_output
        v_feats = []
        batch_feats = []

        for input_id_frames, retrieved_embedding in zip(transform_video_data, retrieved_visual_feature_embedding_video):
            video_feats = []

            for input_id_frame in input_id_frames:
                input_id_frame = input_id_frame.unsqueeze(0).to(self.device)
                retrieved_embedding = retrieved_embedding.to(self.device)

                target_video_frames_embeddings = self.embeddings(input_id_frame)
                target_pre_layernorm_output = self.pre_layernorm(target_video_frames_embeddings)

                video_prompt = self.prompt(target_pre_layernorm_output, retrieved_embedding)
                combined_embeddings = torch.cat([target_pre_layernorm_output, video_prompt], dim=1)

                attention_mask = None
                causal_attention_mask = None

                for layer in self.encoder_layers:
                    layer_output = layer(combined_embeddings, attention_mask, causal_attention_mask)[0]

                outputs = self.post_layernorm(layer_output)

                video_feats.append(outputs)
            video_feats = torch.stack(video_feats, dim=0)
            video_feats = video_feats.squeeze(dim=1)
            batch_feats.append(video_feats)
        v_feats.append(torch.stack(batch_feats))
        video_outputs = torch.cat(v_feats, dim=0).to(self.device)

        return video_outputs
