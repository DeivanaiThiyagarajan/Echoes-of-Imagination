import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToPILImage
import torchvision.transforms as T

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import os
import json
import ast
import gc
import shutil
import glob
import sys
import random
from tqdm import tqdm
from pathlib import Path
from PIL import Image

from text_encoders import clip_model, bert_model

class TextToImageDataset(Dataset):
    def __init__(self, df_combined, transform=None, text_model = 'CLIP'):
        self.data = df_combined
        self.transform = transform
        if(text_model == 'CLIP'):
            self.text_model = clip_model()
        else:
            self.text_model = bert_model()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx, text_model = 'CLIP'):
        row = self.data.iloc[idx]

        embeddings = self.text_model(row['caption'])
        img_name = row['unique_image_identifier'] + ('.jpg' if '.' not in row['unique_image_identifier'] else '')
        img_dir = row['source_dir']
        img_path = img_dir + '/' + img_name
    
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Warning: Image not found at {img_path}")
            image = Image.new('RGB', (256, 256), color='black')
    
        if self.transform:
            image = self.transform(image)
    
        return image, embeddings

class caption_dataset:
    def __init__(self):
        root_directory = os.path.dirname(os.path.dirname(os.getcwd()))
        self.data_dir = root_directory + '/data/datas/extracted_files/'
        self.coco_captions_2014_path = self.data_dir + "annotations_trainval2014/annotations/captions_train2014.json"
        self.coco_captions_2017_path = self.data_dir + "annotations_trainval2017/annotations/captions_val2017.json"

        self.coco_image_dir_2014 = self.data_dir + "train2014/train2014"
        self.coco_image_dir_2017 = self.data_dir + "val2017/val2017"

        self.flickr_annotations_path = self.data_dir + "flickr30k_images/results.csv"
        self.flickr_image_path = self.data_dir + "flickr30k_images/flickr30k_images"

    def load_coco_dataset(self):
        # Load JSON
        with open(self.coco_captions_2014_path, "r") as f:
            captions_2014 = json.load(f)

        with open(self.coco_captions_2017_path, "r") as f:
            captions_2017 = json.load(f)

        # Convert to DataFrames
        df_captions_2014 = pd.DataFrame(captions_2014["annotations"])
        df_captions_2017 = pd.DataFrame(captions_2017["annotations"])

        df_captions_2014['source_dir'] = self.coco_image_dir_2014
        df_captions_2017['source_dir'] = self.coco_image_dir_2017

        df_captions_2014['image_id'] = (
            'COCO_train2014_' + df_captions_2014['image_id'].astype(str).str.zfill(12)
        )

        df_captions_2017['image_id'] = (
            df_captions_2017['image_id'].astype(str).str.zfill(12)
        )

        df_coco_unified = pd.concat([df_captions_2014, df_captions_2017], ignore_index=True)
        df_coco_unified = df_coco_unified[['image_id', 'caption','source_dir']]
        df_coco_unified['source'] = 'COCO'

        # Rename 'image_id' for consistency if you plan to combine with Flickr, 
        # although image access will differ (COCO uses 'image_id' to format the filename).
        df_coco_unified.rename(columns={'image_id': 'unique_image_identifier'}, inplace=True)
        df_coco_unified['unique_image_identifier'] = df_coco_unified['unique_image_identifier'].astype(str)

        return df_coco_unified
    
    def load_flickr_dataset(self):
        try:
            # Use the column names and delimiter identified in our previous steps
            df_flickr = pd.read_csv(
                self.flickr_annotations_path,
                delimiter='|',
                names=['image_name', 'comment_number', 'caption'],
                header=None,
                encoding='utf-8',
                skiprows=1 
            )
            df_flickr['source_dir'] = self.flickr_image_path
            # Flickr image names are already unique strings (e.g., '1000092795.jpg')
            df_flickr = df_flickr[['image_name', 'caption','source_dir']]
            df_flickr.rename(columns={'image_name': 'unique_image_identifier'}, inplace=True)
            df_flickr['source'] = 'Flickr'
        except Exception as e:
            print(f"Error loading Flickr CSV: {e}. Using a small dummy set for Flickr.")
            df_flickr = pd.DataFrame({
                'unique_image_identifier': ['dummy1.jpg', 'dummy2.jpg'],
                'caption': ['A placeholder image.', 'Another example sentence.'],
                'source': 'Flickr'
            })

        return df_flickr

    def load_full_caption_data(self):
        df_flickr = self.load_flickr_dataset()
        df_coco_unified = self.load_coco_dataset()

        df_combined = pd.concat([df_coco_unified, df_flickr], ignore_index=True)

        return df_combined
    
    def get_datasets(self):

        df_combined = self.load_full_caption_data()

        train_df, temp_df = train_test_split(df_combined, test_size=0.2, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

        image_transforms = T.Compose([
            T.Resize((256, 256)), # Target size for the VAE
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_dataset = TextToImageDataset(train_df, image_transforms)
        val_dataset   = TextToImageDataset(val_df, image_transforms)
        test_dataset  = TextToImageDataset(test_df, image_transforms)

        return train_dataset, val_dataset, test_dataset

    def get_dataloader(self, partition, batch_size):
        train_dataset, val_dataset, test_dataset = self.get_datasets()
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        if(partition=='train'):
            return train_loader
        elif(partition == 'val'):
            return val_loader
        return test_loader

