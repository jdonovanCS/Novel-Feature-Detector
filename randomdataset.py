import torch
import cv2
import numpy as np
class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        # self.images = []
        # for i in range(len(self.image_paths)):
        #     image = cv2.imread(self.image_paths[i])
        #     self.images.append(np.transpose(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (2,0,1)))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Maybe load these into memory to speed up
        image_filepath = self.image_paths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.transpose(image, (2, 0, 1))
        # image = self.images[idx]

        label = 'random'
        if self.transform is not None:
            image = self.transform(image=image)['image']

        return image, label