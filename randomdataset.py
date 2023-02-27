import torch
import cv2
import numpy as np
import os
class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = [f for f in os.listdir(self.data_dir) if '.png' in f]
        self.classes = ['random']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Maybe load these into memory to speed up
        image_filepath = os.path.join(self.data_dir, self.images[idx])
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image.astype(np.float32))
        # image = self.images[idx]

        label = 'random'
        if self.transform is not None:
            image = self.transform(image=image)['image']

        return image, 0