import os
import numpy as np
import torch
import torch.utils.data
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

def custom_collate_fn(batch):
    text, item_id, label, \
    retrieved_item_id_list_video, retrieved_visual_feature_embedding_video, retrieved_label_video, \
    retrieved_item_id_list_text, retrieved_text_list_text, retrieved_label_text, \
    transform_video_data = zip(*batch)
    
    return list(text), list(item_id), torch.tensor(label, dtype=torch.float).unsqueeze(1), \
        list(retrieved_item_id_list_video), torch.tensor(retrieved_visual_feature_embedding_video, dtype=torch.float), torch.tensor(retrieved_label_video, dtype=torch.float), \
        list(retrieved_item_id_list_text), list(retrieved_text_list_text), torch.tensor(retrieved_label_text, dtype=torch.float), \
        torch.stack(transform_video_data)

class MyData(Dataset):
    def __init__(self, path, frames_folder_path, frame_num, retrieved_item_num, split):
        super().__init__()
        self.path = path
        self.retrieved_item_num = retrieved_item_num
        self.split = split
        self.frame_nums = frame_num
        self.frames_folder = frames_folder_path
        self.dataframe = pd.read_pickle(path)
        self.text = self.dataframe['text'].tolist()
        self.label_list = self.dataframe['label'].tolist()
        self.item_id_list = self.dataframe['item_id'].tolist()

        
        self.retrieved_item_id_list_video = self.dataframe[f'retrieved_item_id_list_video'].tolist()
        self.retrieved_visual_feature_embedding_video = self.dataframe[f'retrieved_visual_feature_embedding_cls_video'].tolist()
        self.retrieved_label_video = self.dataframe[f'retrieved_label_video'].tolist()
        
        self.retrieved_item_id_list_text = self.dataframe[f'retrieved_item_id_list_text'].tolist()
        self.retrieved_text_list_text = self.dataframe[f'retrieved_text_text'].tolist()
        self.retrieved_label_text = self.dataframe[f'retrieved_label_text'].tolist()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __getitem__(self, index):
        text = self.text[index]
        item_id = self.item_id_list[index]
        label = self.label_list[index]

        retrieved_item_id_list_video = self.retrieved_item_id_list_video[index][:self.retrieved_item_num]
        retrieved_visual_feature_embedding_video = self.retrieved_visual_feature_embedding_video[index][:self.retrieved_item_num]
        retrieved_label_video = self.retrieved_label_video[index][:self.retrieved_item_num]
        
        retrieved_item_id_list_text = self.retrieved_item_id_list_text[index][:self.retrieved_item_num]
        retrieved_text_list_text = self.retrieved_text_list_text[index][:self.retrieved_item_num]
        retrieved_label_text = self.retrieved_label_text[index][:self.retrieved_item_num]

        frame_paths = [os.path.join(self.frames_folder, f"{item_id}_{i}.jpg") for i in range(self.frame_nums)]
        frames = [Image.open(path).convert("RGB") for path in frame_paths]
        transform_video_data = [self.transform(frame) for frame in frames]
        transform_video_data = torch.stack(transform_video_data)
        
        return text, item_id, label, \
            retrieved_item_id_list_video, retrieved_visual_feature_embedding_video, retrieved_label_video, \
                retrieved_item_id_list_text, retrieved_text_list_text, retrieved_label_text, \
                    transform_video_data

    def __len__(self):
        return len(self.dataframe)