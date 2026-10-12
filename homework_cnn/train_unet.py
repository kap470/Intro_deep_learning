
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from data import get_loaders
from models import UNet


# ==================================================
# 1. Configuration
# ==================================================
SEED = 0
BATCH_SIZE = 4
EPOCHS = 100
LEARNING_RATE = 0.001

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Workaround for the cuDNN error in this environment.
torch.backends.cudnn.enabled = False


# ==================================================
# 2. Reproducibility and device
# ==================================================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        images = images.to(device)

        # Segmentation masks: [B, 128, 128]
        masks = targets[1].to(device).long()

        optimizer.zero_grad()

        # Output: [B, 3, 128, 128]
        logits = model(images)

        loss = criterion(logits, masks)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


# ==================================================
# 5. Evaluate segmentation performance
# ==================================================
def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    total_samples = 0
    total_correct_pixels = 0
    total_pixels = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            masks = targets[1].to(device).long()

            logits = model(images)

            loss = criterion(logits, masks)

            # Select the predicted class for each pixel.
            predictions = logits.argmax(dim=1)

            batch_size = images.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            total_correct_pixels += (
                predictions == masks
            ).sum().item()

            total_pixels += masks.numel()

    average_loss = total_loss / total_samples
    pixel_accuracy = total_correct_pixels / total_pixels

    return average_loss, pixel_accuracy


# ==================================================
# 6. Main training procedure
# ==================================================
def main():
    model = UNet(num_classes=3).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    history = []

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        test_loss, pixel_accuracy = evaluate(
            model,
            test_loader,
            criterion
        )

        results = {
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "pixel_accuracy": pixel_accuracy
        }

        history.append(results)

        print(
            f"Epoch [{epoch}/{EPOCHS}] | "
            f"Train loss: {train_loss:.4f} | "
            f"Test loss: {test_loss:.4f} | "
            f"Pixel accuracy: {pixel_accuracy:.2%}"
        )

    model_path = OUTPUT_DIR / "unet.pth"
    torch.save(model.state_dict(), model_path)

    history_path = OUTPUT_DIR / "unet_history.csv"

    with open(history_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "train_loss",
                "test_loss",
                "pixel_accuracy"
            ]
        )
        writer.writeheader()
        writer.writerows(history)

    print("\nTraining completed.")
    print("Saved model:", model_path)
    print("Saved history:", history_path)


if __name__ == "__main__":
    main()
