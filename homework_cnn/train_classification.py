# AI assistance: OpenAI ChatGPT (GPT-6).
# Used to assist with code development, debugging, and explanations.
# https://chatgpt.com/
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from data import get_loaders
from models import SimpleCNN, ResNet18Classifier


# ==================================================
# 1. Configuration
# ==================================================
SEED = 0
BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 0.001

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Workaround for the cuDNN convolution error diagnosed
# in this environment. CUDA remains enabled.
torch.backends.cudnn.enabled = False


# ==================================================
# 2. Reproducibility
# ==================================================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("PyTorch version:", torch.__version__)
print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ==================================================
# 3. Load data
# ==================================================
train_loader, test_loader = get_loaders(
    batch_size=BATCH_SIZE
)

print("Training batches:", len(train_loader))
print("Testing batches:", len(test_loader))


# ==================================================
# 4. Train for one epoch
# ==================================================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    total_loss = 0.0
    total_samples = 0

    for images, targets in loader:
        # Images: [B, 3, 128, 128]
        # targets[0]: binary labels, shape [B]
        # targets[1]: masks, shape [B, 128, 128]

        images = images.to(device)
        labels = targets[0].to(device).long()

        # Clear gradients from the previous batch.
        optimizer.zero_grad()

        # Forward pass: obtain two logits per image.
        logits = model(images)

        # Calculate classification loss.
        loss = criterion(logits, labels)

        # Backpropagation: calculate parameter gradients.
        loss.backward()

        # Update model parameters.
        optimizer.step()

        # Accumulate sample-weighted loss.
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    average_loss = total_loss / total_samples

    return average_loss


# ==================================================
# 5. Evaluate classification performance
# ==================================================
def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # No gradients are needed during evaluation.
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            labels = targets[0].to(device).long()

            logits = model(images)

            loss = criterion(logits, labels)

            # Select the class with the highest logit.
            predictions = logits.argmax(dim=1)

            batch_size = images.size(0)

            total_loss += loss.item() * batch_size
            total_correct += (
                predictions == labels
            ).sum().item()
            total_samples += batch_size

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return average_loss, accuracy


# ==================================================
# 6. Train and evaluate one model
# ==================================================
def run_experiment(model_name, model):
    print("\n" + "=" * 60)
    print("Training:", model_name)
    print("=" * 60)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    history = []

    for epoch in range(1, EPOCHS + 1):

        # Training phase
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer
        )

        # Evaluation phase
        test_loss, test_accuracy = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion
        )

        # Record results for this epoch.
        epoch_results = {
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy
        }

        history.append(epoch_results)

        print(
            f"Epoch [{epoch}/{EPOCHS}] | "
            f"Train loss: {train_loss:.4f} | "
            f"Test loss: {test_loss:.4f} | "
            f"Test accuracy: {test_accuracy:.2%}"
        )

    # --------------------------------------------------
    # Save model parameters
    # --------------------------------------------------
    model_path = OUTPUT_DIR / f"{model_name}.pth"

    torch.save(
        model.state_dict(),
        model_path
    )

    # --------------------------------------------------
    # Save training history as CSV
    # --------------------------------------------------
    history_path = OUTPUT_DIR / f"{model_name}_history.csv"

    with open(history_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "train_loss",
                "test_loss",
                "test_accuracy"
            ]
        )

        writer.writeheader()
        writer.writerows(history)

    print("Saved model:", model_path)
    print("Saved history:", history_path)

    return history


# ==================================================
# 7. Main
# ==================================================
def main():

    # Both models start with independently initialized weights.
    # ResNet-18 is not pretrained: weights=None.
    cnn_history = run_experiment(
        model_name="simple_cnn",
        model=SimpleCNN(num_classes=2)
    )

    resnet_history = run_experiment(
        model_name="resnet18",
        model=ResNet18Classifier(num_classes=2)
    )

    print("\n" + "=" * 60)
    print("CLASSIFICATION EXPERIMENTS COMPLETED")
    print("=" * 60)

    print("CNN final training loss:",
          f"{cnn_history[-1]['train_loss']:.4f}")

    print("ResNet-18 final training loss:",
          f"{resnet_history[-1]['train_loss']:.4f}")


if __name__ == "__main__":
    main()
