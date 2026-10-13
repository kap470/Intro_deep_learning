
import torch
import matplotlib.pyplot as plt

from data import get_loaders
from models import UNet

torch.backends.cudnn.enabled = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the test dataset
_, test_loader = get_loaders(batch_size=16)

# Load the trained U-Net
model = UNet(num_classes=3)
model.load_state_dict(
    torch.load("outputs/unet.pth", map_location=device, weights_only=True)
)
model = model.to(device)
model.eval()

# Display the first three test images
fig, axes = plt.subplots(3, 3, figsize=(10, 10))

images_shown = 0

with torch.no_grad():
    for images, targets in test_loader:
        masks = targets[1]

        images = images.to(device)
        outputs = model(images)
        predicted_masks = outputs.argmax(dim=1).cpu()

        for i in range(images.shape[0]):
            if images_shown == 3:
                break

            # Original image
            image = images[i].cpu().permute(1, 2, 0).numpy()

            # Ground-truth and predicted masks
            ground_truth = masks[i].cpu().numpy()
            prediction = predicted_masks[i].cpu().numpy()

            axes[images_shown, 0].imshow(image)
            axes[images_shown, 0].set_title("Original Image")

            axes[images_shown, 1].imshow(
                ground_truth, cmap="viridis", vmin=0, vmax=2
            )
            axes[images_shown, 1].set_title("Ground Truth")

            axes[images_shown, 2].imshow(
                prediction, cmap="viridis", vmin=0, vmax=2
            )
            axes[images_shown, 2].set_title("U-Net Prediction")

            for j in range(3):
                axes[images_shown, j].axis("off")

            images_shown += 1

        if images_shown == 3:
            break

plt.tight_layout()
plt.savefig("outputs/segmentation_examples.png", dpi=150)
plt.close()
