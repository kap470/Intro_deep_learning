
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from data import get_loaders
from models import UNet


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------
BATCH_SIZE = 4
NUM_IMAGES = 3

MODEL_PATH = Path("outputs/unet.pth")
OUTPUT_PATH = Path("outputs/unet_predictions.png")

# Same workaround used during training.
torch.backends.cudnn.enabled = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# --------------------------------------------------
# 2. Load the trained model
# --------------------------------------------------
model = UNet(num_classes=3).to(device)

state_dict = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(state_dict)
model.eval()

print("Loaded model:", MODEL_PATH)


# --------------------------------------------------
# 3. Load test images and masks
# --------------------------------------------------
_, test_loader = get_loaders(batch_size=BATCH_SIZE)

images, targets = next(iter(test_loader))

# Classification labels are targets[0].
# Segmentation masks are targets[1].
masks = targets[1]

images = images.to(device)
masks = masks.to(device)


# --------------------------------------------------
# 4. Predict segmentation masks
# --------------------------------------------------
with torch.no_grad():
    logits = model(images)

    # [B, 3, 128, 128] -> [B, 128, 128]
    predictions = logits.argmax(dim=1)

images = images.cpu()
masks = masks.cpu().numpy()
predictions = predictions.cpu().numpy()


# --------------------------------------------------
# 5. Plot original images and masks
# --------------------------------------------------
class_names = ["Pet", "Background", "Boundary"]

num_images = min(NUM_IMAGES, images.shape[0])

fig, axes = plt.subplots(
    num_images, 3,
    figsize=(12, 4 * num_images),
    squeeze=False
)

for i in range(num_images):
    # Convert image from [C, H, W] to [H, W, C].
    image = images[i].permute(1, 2, 0).numpy()

    ground_truth = masks[i]
    prediction = predictions[i]

    axes[i, 0].imshow(image)
    axes[i, 0].set_title("Original image")

    axes[i, 1].imshow(
        ground_truth,
        cmap="viridis",
        vmin=0,
        vmax=2,
        interpolation="nearest"
    )
    axes[i, 1].set_title("Ground-truth mask")

    axes[i, 2].imshow(
        prediction,
        cmap="viridis",
        vmin=0,
        vmax=2,
        interpolation="nearest"
    )
    axes[i, 2].set_title("U-Net prediction")

    for ax in axes[i]:
        ax.axis("off")

fig.suptitle(
    "U-Net Segmentation Results",
    fontsize=16
)

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
print("Saved figure:", OUTPUT_PATH)

plt.show()
