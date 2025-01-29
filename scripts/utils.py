import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader

import json 


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


def load_model(model_path, model_class):
    """
    Load a PyTorch model from a saved state dictionary.

    Args:
        model_path (str): Path to the model `.pth` file.
        model_class (nn.Module): The model class definition to reconstruct the architecture.

    Returns:
        model (nn.Module): The loaded PyTorch model.
    """
    # Instantiate the model
    model = model_class()
    # Load the state dictionary
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model


def load_metadata(metadata_path):
    """
    Load metadata from a JSON file.

    Args:
        metadata_path (str): Path to the metadata `.json` file.

    Returns:
        metadata (dict): The metadata dictionary.
    """
    with open(metadata_path, "r") as metadata_file:
        metadata = json.load(metadata_file)
    return metadata


def add_metadata(metadata_path, params):
    """
    Add or update metadata in a JSON file.

    Args:
        metadata_path (str): Path to the metadata `.json` file.
        params (dict): The metadata to add to the file.
    """
    metadata = {}
    try:

        with open(metadata_path, "r") as metadata_file:
            metadata = json.load(metadata_file)
    except FileNotFoundError:
        pass  
    except json.JSONDecodeError:
        raise ValueError(f"Le fichier {metadata_path} n'est pas un JSON valide.")

    metadata.update(params)

    with open(metadata_path, "w") as metadata_file:
        json.dump(metadata, metadata_file, indent=4)


def check_if_metadata_has_results(metadata_path):
    """
    Check if a metadata JSON file contains a 'results' key.

    Args:
        metadata_path (str): Path to the metadata `.json` file.

    Returns:
        bool: True if the 'results' key exists, False otherwise.
    """
    try:
        with open(metadata_path, "r") as metadata_file:
            metadata = json.load(metadata_file)
        return "Accuracy" in metadata
    except FileNotFoundError:
        return False  
    except json.JSONDecodeError:
        raise ValueError(f"The file {metadata_path} is not a valid JSON")
