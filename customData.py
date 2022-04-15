
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset


class CustomData(Dataset):
    def __init__(self, image_paths, transform=False) -> None:
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx) -> tuple:
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path, 0)
        image = cv2.threshold(image, 127, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)[
            1]
        image = cv2.resize(image, (28, 28))
        image = image.reshape(28, 28, 1)
        image = image.astype(np.float32)
        image = image / 255.0

        label = int(image_path.split('/')[-1].split('.')[0])

        if self.transform:
            image = self.transform(image)

        return image, label
