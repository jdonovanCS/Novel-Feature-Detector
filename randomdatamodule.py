from typing import Any, Callable, Optional, Sequence, Union
from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl
from torchvision import transforms
import randomdataset
import os
import cv2
import numpy as np

class RandomDataModule(pl.LightningDataModule):

    name = "random"
    # dataset_cls = CIFAR100
    dims = (3, 32, 32)

    def __init__(self,
        data_dir: Optional[str] = None,
        val_split: Union[int, float] = 0.2,
        transform = None,
        num_workers: int = 0,
        normalize: bool = False,
        batch_size: int = 32,
        seed: int = 42,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:

        """
        Args:
            data_dir: Where to save/load the data
            val_split: Percent (float) or number (int) of samples to use for the validation split
            num_workers: How many workers to use for loading data
            normalize: If true applies image normalize
            batch_size: How many samples per batch to load
            seed: Random seed to be used for train/val/test splits
            shuffle: If true shuffles the train data every epoch
            pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before
                        returning them
            drop_last: If true drops the last incomplete batch
        """

        super(RandomDataModule).__init__(  # type: ignore[misc]
            data_dir=data_dir,
            val_split=val_split,
            transform = None,
            num_workers=num_workers,
            normalize=normalize,
            batch_size=batch_size,
            seed=seed,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last,
            *args,
            **kwargs,
        )

        self.data_dir = data_dir
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers

    @property
    def num_classes(self) -> int:
        """
        Return:
            100
        """
        return 1

    def default_transforms(self) -> Callable:
        stats = ((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025))
        transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32,padding=4,padding_mode="reflect"),
        transforms.ToTensor(),
        transforms.Normalize(*stats)
        ])
        return transform
    
    def prepare_data(self):
        # download data to data directory
        num_images = len([f for f in os.listdir(self.data_dir) if '.jpg' in f])
        if num_images < self.batch_size:
            for i in range(num_images, self.batch_size):
                rgb = np.random.randint(255, size=(32,32,3), dtype=np.uint8)
                cv2.imwrite('images/random/{}.png'.format(i), rgb)
    
    def setup(self, stage=None):
        self.random_full = randomdataset(self.data_dir, transform=self.transform)
            
    def train_dataloader(self):
        return DataLoader(self.random_full, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.random_full, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.random_full, batch_size=self.batch_size, num_workers=self.num_workers)