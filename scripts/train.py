import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader


def train_model(model, train_loader, criterion, optimizer, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")


import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import uuid
import json
from datetime import datetime


def train_model_and_save(
    model,
    train_loader,
    criterion,
    optimizer,
    epochs,
    learning_rate, 
    model_dir= "models\\saved_models",
    metadata_dir="models\\saved_models_metadata",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Ensure directories exist
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    for epoch in range(epochs):

        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Logging epoch loss
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss}")

    # Save model and metadata after training
    unique_id = str(uuid.uuid4())
    model_path = os.path.join(model_dir, f"model_{unique_id}.pth")
    metadata_path = os.path.join(metadata_dir, f"metadata_{unique_id}.json")

    # Save model state dictionary
    torch.save(model.state_dict(), model_path)

    # Save metadata including model architecture
    model_config = {
        "class_name": model.__class__.__name__,
        "state_dict": {
            k: v.shape for k, v in model.state_dict().items()
        },  # Parameter shapes
        "model_parameters": str(model),  # String representation for reference
    }

    metadata = {
        "id": unique_id,
        "timestamp": datetime.now().isoformat(),
        "epochs": epochs,
        "final_loss": avg_loss,
        "device": device.type,
        "model_path": model_path,
        "optimizer": optimizer.__class__.__name__,
        "learning_rate": learning_rate,
        "model_architecture": model_config,
    }
    with open(metadata_path, "w") as metadata_file:
        json.dump(metadata, metadata_file, indent=4)

    print(f"Model saved to {model_path}")
    print(f"Metadata saved to {metadata_path}")
