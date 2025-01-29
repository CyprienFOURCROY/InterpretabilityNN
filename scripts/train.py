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


import torch
import os
import uuid
import json
from datetime import datetime


import torch
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
    model_dir="models\\saved_models",
    metadata_dir="models\\saved_models_metadata",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    # Generate a unique ID for the training session
    parent_id = str(uuid.uuid4())
    metadata_path = os.path.join(metadata_dir, f"metadata_{parent_id}.json")
    metadata = {
        "parent_id": parent_id,
        "timestamp": datetime.now().isoformat(),
        "epochs": epochs,
        "device": device.type,
        "optimizer": optimizer.__class__.__name__,
        "learning_rate": learning_rate,
        "model_architecture": {
            "class_name": model.__class__.__name__,
            "state_dict_shapes": {k: v.shape for k, v in model.state_dict().items()},
            "model_parameters": str(model),
        },
        "epoch_details": [],
    }

    initial_model_path = os.path.join(model_dir, f"model_{parent_id}_epoch0.pth")
    torch.save(model.state_dict(), initial_model_path)
    metadata["epoch_details"].append(
        {
            "id": parent_id,
            "model_path": initial_model_path,
            "epoch": 0,
            "loss": None,  # No loss for the initial weights
        }
    )
    print(f"Saved initial random weights at Epoch 0 to {initial_model_path}")

    for epoch in range(1, epochs + 1):
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

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss}")

        # Save the model for this epoch
        epoch_id = str(uuid.uuid4())
        model_path = os.path.join(model_dir, f"model_{epoch_id}_epoch{epoch}.pth")
        torch.save(model.state_dict(), model_path)

        # Append epoch details to metadata
        metadata["epoch_details"].append(
            {
                "id": epoch_id,
                "model_path": model_path,
                "epoch": epoch,
                "loss": avg_loss,
            }
        )

    # Save metadata to a JSON file
    with open(metadata_path, "w") as metadata_file:
        json.dump(metadata, metadata_file, indent=4)

    print(f"Training complete. Metadata saved to {metadata_path}.")
