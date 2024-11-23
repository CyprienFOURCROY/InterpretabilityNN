import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class MNISTCSV(Dataset):
    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the csv file with images and labels.
        """
        self.data = pd.read_csv(csv_file)
        self.labels = self.data["label"].values  # Extract the labels
        self.images = self.data.drop("label", axis=1).values  # Extract the pixel data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the image as a flattened vector and reshape it into (1, 28, 28)
        image = (
            self.images[idx].reshape(28, 28).astype("float32") / 255.0
        )  # Normalize the pixel values
        label = self.labels[idx]

        # Convert to torch tensors
        image_tensor = torch.tensor(image).unsqueeze(
            0
        )  # Add channel dimension (1, 28, 28)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return image_tensor, label_tensor

