import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class SelfDrivingCarDataset(Dataset):
    w, h = 224, 224

    def __init__(self, df, image_dir):
        self.image_dir = image_dir
        self.df = df
        self.files = glob.glob(os.path.join(self.image_dir, '*.jpg'))
        self.image_infos = df.frame.unique()
        print(f"Dataset initialized with {len(self.image_infos)} images.")

    def __len__(self):
        return len(self.image_infos)

    def __getitem__(self, ix):
        img_id = self.image_infos[ix]
        img_path = os.path.join(self.image_dir, img_id)
        img = Image.open(img_path).convert('RGB')
        img = np.array(img.resize((self.w, self.h), resample=Image.BILINEAR)) / 255.
        data = self.df[self.df['frame'] == img_id]
        labels = data['class_id'].values.tolist()
        data = data[['xmin', 'ymin', 'xmax', 'ymax']].values
        data[:, [0, 2]] *= self.w
        data[:, [1, 3]] *= self.h
        boxes = data.astype(np.uint32).tolist()
        target = {}
        target["boxes"] = torch.Tensor(boxes).float()
        target["labels"] = torch.Tensor([i for i in labels]).long()
        img = self.preprocess_image(img)
        return img, target

    def preprocess_image(self, img):
        img = torch.tensor(img).permute(2, 0, 1)
        return img.float()

    def collate_fn(self, batch):
        images = []
        targets = []
        for img, target in batch:
            images.append(img)
            targets.append(target)
        return images, targets
