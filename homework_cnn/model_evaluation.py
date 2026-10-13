# AI assistance: OpenAI ChatGPT (GPT-6).
# Used to assist with code development, debugging, and explanations.
# https://chatgpt.com/
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

from data import get_loaders
from models import SimpleCNN, ResNet18Classifier, UNet


# Device
torch.backends.cudnn.enabled = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load test data
_, test_loader = get_loaders(batch_size=16)


# --------------------------------------------------
# 1. Evaluate CNN and ResNet-18
# --------------------------------------------------
def evaluate_classifier(model, model_path):
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model = model.to(device)
    model.eval()

    predictions = []
    true_labels = []

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            labels = targets[0].to(device)

            outputs = model(images)
            predicted_labels = outputs.argmax(dim=1)

            predictions.extend(predicted_labels.cpu().tolist())
            true_labels.extend(labels.cpu().tolist())

    accuracy = accuracy_score(true_labels, predictions)

    return accuracy, true_labels


cnn_accuracy, test_labels = evaluate_classifier(
    SimpleCNN(num_classes=2), "outputs/simple_cnn.pth"
)

resnet_accuracy, _ = evaluate_classifier(
    ResNet18Classifier(num_classes=2), "outputs/resnet18.pth"
)

# majority-class baseline
majority_baseline = sum(label == 1 for label in test_labels) / len(test_labels)

print("\nClassification Results")
print("| Model | Test accuracy |")
print("|---|---:|")
print(f"| Regular CNN | {cnn_accuracy:.2%} |")
print(f"| ResNet-18 | {resnet_accuracy:.2%} |")
print(f"| Always-Dog baseline | {majority_baseline:.2%} |")


# --------------------------------------------------
# 2. Evaluate U-Net pixel accuracy
# --------------------------------------------------
model = UNet(num_classes=3)

model.load_state_dict(
    torch.load("outputs/unet.pth", map_location=device, weights_only=True)
)
model = model.to(device)
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, targets in test_loader:
        images = images.to(device)
        masks = targets[1].to(device).long()

        outputs = model(images)
        predicted_masks = outputs.argmax(dim=1)

        correct += (predicted_masks == masks).sum().item()
        total += masks.numel()

pixel_accuracy = correct / total

print("\nU-Net Segmentation Results")
print(f"Pixel accuracy: {pixel_accuracy:.2%}")
