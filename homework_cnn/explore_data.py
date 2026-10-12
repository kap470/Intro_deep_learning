
# AI assistance: OpenAI ChatGPT (GPT-6).
# Used to assist with code development, debugging, and explanations.
# https://chatgpt.com/
import numpy as np
import matplotlib.pyplot as plt

from data import get_raw_datasets, get_loaders, CLASS_NAMES

# ==================================================
# 1. Explore the raw datasets
# ==================================================
train_dataset, test_dataset = get_raw_datasets()

print("DATASET INFORMATION")
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of testing samples:  {len(test_dataset)}")

# Inspect the first raw training sample
image, (label, mask) = train_dataset[0]

print("\nFIRST RAW TRAINING SAMPLE")
print(f"Image dimensions (width, height): {image.size}")
print(f"Classification label: {label} ({CLASS_NAMES[label]})")
print(f"Mask dimensions (width, height): {mask.size}")
print(f"Unique mask values: {np.unique(np.array(mask))}")

# ==================================================
# 2. Visualize the first three training images
#    beside their segmentation masks
# ==================================================
fig, axes = plt.subplots(3, 2, figsize=(10, 12))

for i in range(3):
    image, (label, mask) = train_dataset[i]

    axes[i, 0].imshow(image)
    axes[i, 0].set_title(
        f"Training image {i}: {CLASS_NAMES[label]}"
    )
    axes[i, 0].axis("off")

    # Original masks contain values 1, 2, 3.
    axes[i, 1].imshow(
        np.array(mask),
        cmap="viridis",
        vmin=1,
        vmax=3,
        interpolation="nearest"
    )
    axes[i, 1].set_title(f"Segmentation mask {i}")
    axes[i, 1].axis("off")

fig.suptitle("Oxford-IIIT Pet Dataset: Raw Training Samples",
             fontsize=14)
plt.tight_layout()

# Save the figure for the README
plt.savefig("dataset_samples.png", dpi=200, bbox_inches="tight")
plt.show()

print("\nSaved figure: dataset_samples.png")

# ==================================================
# 3. Inspect one processed batch
# ==================================================
train_loader, test_loader = get_loaders(batch_size=16)

images, (labels, masks) = next(iter(train_loader))

print("\nPROCESSED TRAINING BATCH")
print(f"Images shape: {images.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Masks shape:  {masks.shape}")
print(f"Image data type: {images.dtype}")
print(f"Label data type: {labels.dtype}")
print(f"Mask data type:  {masks.dtype}")
print(f"Processed mask values: {np.unique(masks.numpy())}")
