import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split
import pandas as pd

from .dataset import radImageDataset


class datasetPrepper:
    def __init__(
        self,
        dataframe_path,
        image_dir="data/test_images",
        test_split=0.2,
        random_state=43,
        batch_size=32,
    ):
        self.df = pd.read_csv(dataframe_path)
        self.image_dir = image_dir
        self.test_split = test_split
        self.random_state = random_state
        self.batch_size = batch_size

        self.train_dataset = None
        self.val_dataset = None
        self.train_loader = None
        self.val_loader = None
        self.class_names = None
        self.class_weights = None

    def prepare(self, compute_class_weights=False):
        train_transform = self.get_train_transform()
        val_transform = self.get_val_transform()

        train_df, val_df = train_test_split(
            self.df,
            test_size=self.test_split,
            stratify=self.df["label"],
            random_state=self.random_state,
        )

        self.train_dataset = radImageDataset(
            train_df, self.image_dir, transform=train_transform
        )
        self.val_dataset = radImageDataset(
            val_df, self.image_dir, transform=val_transform
        )

        self.class_names = self.train_dataset.class_names

        if compute_class_weights:
            self._compute_class_weights()

        self._create_loaders()

        return self

    def get_train_transform(self):
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
            ]
        )

    def get_val_transform(self):
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )

    def _compute_class_weights(self):
        counts = self.df["label"].value_counts().sort_index().values
        self.class_weights = 1.0 / torch.tensor(counts, dtype=torch.float)
        self.class_weights = self.class_weights / self.class_weights.sum() * len(counts)

    def _create_loaders(self):
        if self.class_weights is not None:
            sample_weights = [
                self.class_weights[label].item() for label in self.train_dataset.labels
            ]
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=4,
                pin_memory=True,
            )
        else:
            self.train_loader = DataLoader(
                self.train_dataset, batch_size=self.batch_size, shuffle=True
            )

        self.val_loader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False
        )
