import copy
from typing import Union, List, Tuple, Dict, Optional
from angle_emb import AnglE
import torch.nn as nn
import numpy as np
import torch

from models.prompt import Text_Prompt_Generator
class AnglEModel_ICPF(torch.nn.Module):
    def __init__(self, angle_model_path, prompt_re_length, retrieved_num: int, device, max_length = 512):
        """
        Initialize the AnglEModel_ICPF.

        Args:
            angle_model_path (str): Path to the pre-trained AnglE model.
            prompt_re_length (int): Length of the prompt.
            retrieved_num (int): Number of retrieved instances.
            device (str): Device to run the model on ('cpu' or 'cuda').
            max_length (int, optional): Maximum length of input sequences. Defaults to 512.
        """
        super(AnglEModel_ICPF, self).__init__()
        self.device = device
        self.feature_dim = 768
        self.prompt_re_length = prompt_re_length
        angle_model = AnglE.from_pretrained(angle_model_path)
        self.tokenizer = angle_model.tokenizer
        self.backbone = angle_model.backbone
        self.backbone_embedding_layer = self.backbone.embeddings
        self.backbone_encoder_layer = self.backbone.encoder

        self.prompt_layer = Text_Prompt_Generator(prompt_re_length, device=device)
        self.max_length = max_length
        self.retrieved_num = retrieved_num

        backbone_bert_layers = (self.backbone_encoder_layer.layer).to(device)
        self.backbone_bert_layers = nn.ModuleList(backbone_bert_layers)

        self.freeze(self.backbone)
        self.freeze(self.backbone_embedding_layer)
        self.freeze(self.backbone_encoder_layer)

    def freeze(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def encode(self,
               text: Union[List[str], Tuple[str], List[Dict], str],
               retrieved_texts: List[List[str]],
               max_length: Optional[int] = None,
               end_with_eos: bool = False,
               layer_index: int = -1
               ):
        """
        Encode the input text and retrieved texts.

        Args:
            text: Input text to encode.
            retrieved_texts: List of retrieved texts for each input text.
            max_length: Maximum length of input sequences.
            end_with_eos: Whether to end sequences with EOS token.
            layer_index: Index of the layer to use for encoding (-1 for last layer).

        Returns:
            torch.Tensor: Encoded text output.
        """
        global text_output
        if layer_index != -1 and self.full_backbone is None:
            self.full_backbone = copy.deepcopy(self.backbone)

        if layer_index != -1:
            self.backbone.encoder.layer = self.full_backbone.encoder.layer[:layer_index]

        with torch.no_grad():
            if not isinstance(text, (tuple, list)):
                text = [text]
            max_length = max_length or self.max_length
            if end_with_eos:
                max_length -= 1

            tok_text = self.tokenizer(
                text,
                padding='max_length',
                max_length=max_length,
                truncation=True,
                return_tensors='pt').to(self.device)

            batch_size = tok_text['attention_mask'].size(0)
            extended_attention_mask = torch.cat([tok_text['attention_mask'],torch.ones(batch_size, self.prompt_re_length).to(self.device)], dim=1)
            extended_attention_mask = extended_attention_mask.unsqueeze(1).unsqueeze(2)

            target_text_embeddings = self.backbone_embedding_layer(input_ids=tok_text['input_ids'], token_type_ids=tok_text['token_type_ids'])

            retrieved_text_embeddings = []
            if retrieved_texts:
                for rt_list in retrieved_texts:
                    tok_retrieved = self.tokenizer(
                        rt_list,
                        padding='max_length',
                        max_length=max_length,
                        truncation=True,
                        return_tensors='pt').to(self.device)
                    for i in range(len(tok_retrieved['input_ids'])):
                        retrieved_text_middle_output = self.backbone_embedding_layer(
                            input_ids=tok_retrieved['input_ids'][i:i+1],
                            token_type_ids=tok_retrieved['token_type_ids'][i:i+1]
                        )
                        retrieved_text_embeddings.append(retrieved_text_middle_output)

            retrieved_text_embeddings = torch.stack(retrieved_text_embeddings)
            retrieved_text_embeddings = torch.reshape(retrieved_text_embeddings, (target_text_embeddings.size(0), self.retrieved_num, 512, 768))


        text_prompt = self.prompt_layer(target_text_embeddings, retrieved_text_embeddings)
        aggregated_text_embeddings = torch.cat([target_text_embeddings, text_prompt], dim=1)

        for layer in self.backbone_bert_layers:
            text_output = layer(aggregated_text_embeddings, attention_mask=extended_attention_mask)[0]

        return text_output
