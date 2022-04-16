import cv2
import numpy as np
from torch.utils.data import Dataset


class CustomData(Dataset):
    """A custom data set class inheriting from the torch.utils.data.Dataset
    class used to create a dataset from custom images."""

    def __init__(self, image_paths, transform=False) -> None:
        """The constructor for defining image paths and the transform.

        Args:
            image_paths (list): A list of image paths.
            transform (bool): A boolean value to determine whether to apply a
                transform to the image.
        """
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        """Returns the length of the dataset.

        Returns:
            int: length of the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx) -> tuple:
        """Returns the item at the given index.

        Args:
            idx (int): The index of the item to return.

        Returns:
            image: The item at the given index.
            label: The target of the item at the given index.
        """
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
