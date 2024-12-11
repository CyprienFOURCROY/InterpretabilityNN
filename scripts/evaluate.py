import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def evaluate_model(model, test_loader, device):

    model.eval()
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            # Collect true and predicted labels for confusion matrix
            true_labels.extend(
                labels.cpu().numpy()
            )  # Move labels to CPU and convert to NumPy
            pred_labels.extend(
                predicted.cpu().numpy()
            )  # Move predicted labels to CPU and convert to NumPy

    # Generate the confusion matrix
    try:
        cm = confusion_matrix(true_labels, pred_labels)
        print(cm)
    except ValueError as e:
        print(f"Error in confusion matrix calculation: {e}")
        return None  # Handle error more gracefully by returning None

    # Print test accuracy
    accuracy = np.trace(cm) / np.sum(cm)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Plot confusion matrix using seaborn
    if cm is not None:
        plt.figure(figsize=(10, 7))
        sns.heatmap(
            cm,
            annot=True,
            fmt="g",
            cmap="Blues",
            xticklabels=range(10),
            yticklabels=range(10),
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.show()

    return cm


from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def evaluate_model_v2(model, test_loader, device):
    """
    Evaluate a model on the test dataset and calculate F1, accuracy, recall, and precision.

    Args:
        model: The PyTorch model to evaluate.
        test_loader: DataLoader for the test dataset.
        device: Device to use (e.g., 'cpu' or 'cuda').

    Returns:
        metrics (dict): A dictionary containing F1, accuracy, recall, and precision.
    """
    model.eval()
    true_labels = []
    pred_labels = []

    # Disable gradient calculation for evaluation
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            # Collect true and predicted labels
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predicted.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average="macro")
    recall = recall_score(true_labels, pred_labels, average="macro")
    f1 = f1_score(true_labels, pred_labels, average="macro")

    # Store metrics in a dictionary
    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
    }

    # Print the metrics
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")

    return metrics
