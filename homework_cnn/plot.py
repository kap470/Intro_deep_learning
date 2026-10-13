
import pandas as pd
import matplotlib.pyplot as plt

# Read training histories
cnn = pd.read_csv("outputs/simple_cnn_history.csv")
resnet = pd.read_csv("outputs/resnet18_history.csv")
unet = pd.read_csv("outputs/unet_history.csv")

# Plot CNN and ResNet-18 training losses together
plt.figure()
plt.plot(cnn["epoch"], cnn["train_loss"], label="Regular CNN")
plt.plot(resnet["epoch"], resnet["train_loss"], label="ResNet-18")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.legend()
plt.savefig("outputs/classification_training_loss.png")
plt.close()

# Plot U-Net training loss separately
plt.figure()
plt.plot(unet["epoch"], unet["train_loss"], label="U-Net")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.legend()
plt.savefig("outputs/unet_training_loss.png")
plt.close()
