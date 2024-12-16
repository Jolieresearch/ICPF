import gc
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from angle_emb import AnglE
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm

device = "cuda:0" if torch.cuda.is_available() else "cpu"

clip_model = CLIPModel.from_pretrained("clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("clip-vit-base-patch32")
image_encoder = clip_model.vision_model

angle = AnglE.from_pretrained('angle-bert-base-uncased-nli-en-v1', pooling_strategy='cls').to(device)

def load_video_frames(item_id, frame_dir, num_frames):
    frames = []
    for i in range(num_frames):
        frame_path = os.path.join(frame_dir, f"{item_id}_{i}.jpg")
        if os.path.exists(frame_path):
            frames.append(Image.open(frame_path))
        else:
            print(f"Warning: Frame {frame_path} not found.")
    return frames

class VideoTextDataset(Dataset):
    def __init__(self, df, frame_dir, num_frames):
        self.df = df
        self.frame_dir = frame_dir
        self.num_frames = num_frames

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        frames = load_video_frames(item['item_id'], self.frame_dir, self.num_frames)

        video_features = process_frames_batch(frames, image_encoder, clip_processor, self.num_frames)
        text_features = process_text_batch(item['text'])

        return {
            'id': item['item_id'],
            'video_features': video_features,
            'text_features': text_features
        }

def process_frames_batch(frames_batch, image_encoder, processor, num_frames_per_video):
    inputs = processor(images=frames_batch, return_tensors="pt", padding=True)
    pixel_values = inputs.pixel_values.to(device)

    with torch.no_grad():
        outputs = image_encoder(pixel_values=pixel_values)

    image_features = outputs.last_hidden_state[:, 0, :].cpu().numpy()

    # Reshape to group frames by video
    image_features = image_features.reshape(-1, num_frames_per_video, image_features.shape[-1])

    # Calculate mean feature for each video
    mean_features = np.mean(image_features, axis=1)

    del inputs, pixel_values, outputs
    torch.cuda.empty_cache()

    return mean_features

def process_text_batch(texts):
    with torch.no_grad():
        text_features = angle.encode(texts, to_numpy=True)
    torch.cuda.empty_cache()
    return text_features

def extract_features_and_save(df, frame_dir, num_frames, output_path):
    dataset = VideoTextDataset(df, frame_dir, num_frames)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False, collate_fn=lambda x: x)

    all_video_features = []
    all_text_features = []
    all_item_ids = []

    for batch in tqdm(dataloader, desc="Extracting features"):
        for sample in batch:
            all_video_features.append(sample['video_features'])
            all_text_features.append(sample['text_features'])
            all_item_ids.append(sample['id'])

    # 将特征与 item_id 关联
    all_video_features = np.concatenate(all_video_features, axis=0)
    all_text_features = np.concatenate(all_text_features, axis=0)

    df['video_features'] = list(all_video_features)
    df['text_features'] = list(all_text_features)
    df['item_id'] = all_item_ids

    df.to_pickle(output_path)

def main_feature_extraction(train_path, valid_path, test_path, frame_dir, num_frames):
    train_df = pd.read_pickle(train_path)
    valid_df = pd.read_pickle(valid_path)
    test_df = pd.read_pickle(test_path)

    extract_features_and_save(train_df, frame_dir, num_frames, train_path.replace('.pkl', '_features.pkl'))
    extract_features_and_save(valid_df, frame_dir, num_frames, valid_path.replace('.pkl', '_features.pkl'))
    extract_features_and_save(test_df, frame_dir, num_frames, test_path.replace('.pkl', '_features.pkl'))


if __name__ == "__main__":
    train_path = 'train.pkl'
    valid_path = 'valid.pkl'
    test_path = 'test.pkl'
    frame_dir = r'microlens\video_frames'
    num_frames = 10
    main_feature_extraction(train_path, valid_path, test_path, frame_dir, num_frames)
