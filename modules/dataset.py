import torch
from torch.utils.data import Dataset
from PIL import Image
import os


class radImageDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None, class_names=None):
        self.df = dataframe
        self.image_dir = image_dir
        self.transform = transform

        if class_names is None:
            self.class_names = self.df["label"].astype("category").cat.categories
        else:
            self.class_names = class_names

        label_to_idx = {name: i for i, name in enumerate(self.class_names)}
        self.labels = self.df["label"].map(label_to_idx).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.df.iloc[idx]["filename"])
        image = Image.open(img_path).convert("RGB")
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label
