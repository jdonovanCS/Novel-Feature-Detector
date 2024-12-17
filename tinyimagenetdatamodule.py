import os
from typing import Optional
import pytorch_lightning as pl
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from tinyimagenetdataset import TinyImageNetDataset

class TinyImageNetDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "tiny-imagenet-200", batch_size: int = 64, num_workers: int = 4, pin_memory=False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_transform = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    @property
    def num_classes(self) -> int:
        """
        Return:
            100
        """
        return 200
    
    def prepare_data(self):
        # Download the dataset if not present
        if not os.path.exists(self.data_dir):
            # Download the dataset (you'll need to implement this part)
            TinyImageNetDataset(self.data_dir, download=True)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            train_dataset = TinyImageNetDataset(self.data_dir, transform=self.train_transform, mode='train')
            val_dataset = TinyImageNetDataset(self.data_dir, transform=self.val_transform, mode='val')

            self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if stage == "test" or stage is None:
            self.test_dataset = TinyImageNetDataset(self.data_dir, transform=self.val_transform, mode='val')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)