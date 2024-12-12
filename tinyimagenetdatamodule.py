import pytorch_lightning as pl
from torchvision import datasets, transforms

class TinyImageNetDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=64, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        # Download the dataset if it doesn't exist
        datasets.ImageFolder(self.data_dir + "/train")

    def setup(self, stage=None):
        # Create the train, val, and test datasets
        train_transform = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        self.train_dataset = datasets.ImageFolder(self.data_dir + "/train", transform=train_transform)
        self.val_dataset = datasets.ImageFolder(self.data_dir + "/val", transform=val_transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )