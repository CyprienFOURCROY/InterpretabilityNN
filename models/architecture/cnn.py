import torch
import torch.nn as nn
import torch.nn.functional as F


class FashionMNISTCNN(nn.Module):
    def __init__(self, hidden_layers=[128], freeze_cnn=True):
        super(FashionMNISTCNN, self).__init__()
        # CNN layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dynamically create fully connected layers based on the `hidden_layers` argument
        self.fc_layers = nn.ModuleList()
        input_size = 64 * 7 * 7  # Flattened input from CNN

        for hidden_size in hidden_layers:
            self.fc_layers.append(nn.Linear(input_size, hidden_size))
            input_size = hidden_size  # Set input size for the next layer

        # Final output layer (fixed for Fashion MNIST with 10 classes)
        self.output_layer = nn.Linear(input_size, 10)

        # Freeze CNN layers if required
        if freeze_cnn:
            self.freeze_cnn_layers()

    def forward(self, x):
        # Forward pass through CNN layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # Flatten

        # Forward pass through dynamically created fully connected layers
        for fc in self.fc_layers:
            x = F.relu(fc(x))

        # Final output layer
        x = self.output_layer(x)
        return x

    def freeze_cnn_layers(self):
        """Freeze the CNN layers so they don't get updated during training."""
        for param in self.conv1.parameters():
            param.requires_grad = False
        for param in self.conv2.parameters():
            param.requires_grad = False
