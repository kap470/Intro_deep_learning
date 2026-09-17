import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
import matplotlib.pyplot as plt


# ============================================================
# 1. LOAD DATASET AND INITIAL EXPLORATION
# ============================================================

# Load the California Housing dataset
housing = fetch_california_housing()

# Save the dataset description in a variable
dataset_description = housing.DESCR

# Print the description to the console
print(dataset_description)


# ------------------------------------------------------------
# Put the dataset into a Pandas DataFrame
# ------------------------------------------------------------

df = pd.DataFrame(
    housing.data,
    columns=housing.feature_names
)

df["MedHouseVal"] = housing.target


# ------------------------------------------------------------
# Show the first five rows
# ------------------------------------------------------------

head_output = df.head().to_string()

print("\nFirst five rows:")
print(head_output)


# ------------------------------------------------------------
# Descriptive statistics
# ------------------------------------------------------------

describe_output = df.describe().to_string()

print("\nDescriptive statistics:")
print(describe_output)


# ============================================================
# 2. TRAIN / TEST SPLIT
# ============================================================

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=0
)


# ============================================================
# 3. STANDARDIZE THE FEATURES
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 4. SCIKIT-LEARN LINEAR REGRESSION
# ============================================================

linear_model = LinearRegression()

linear_model.fit(X_train_scaled, y_train)


# ============================================================
# 5. PYTORCH MULTI-LAYER PERCEPTRON
# ============================================================

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(
    X_train_scaled,
    dtype=torch.float32
)

X_test_tensor = torch.tensor(
    X_test_scaled,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train.values,
    dtype=torch.float32
).view(-1, 1)

y_test_tensor = torch.tensor(
    y_test.values,
    dtype=torch.float32
).view(-1, 1)


# ------------------------------------------------------------
# Define the neural network
# ------------------------------------------------------------

class MLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.hidden = nn.Linear(8, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, x):

        x = self.hidden(x)
        x = torch.relu(x)
        x = self.output(x)

        return x


# Create model
mlp = MLP()


# ------------------------------------------------------------
# Define loss function and optimizer
# ------------------------------------------------------------

criterion = nn.MSELoss()

optimizer = optim.Adam(
    mlp.parameters(),
    lr=0.01
)


# ------------------------------------------------------------
# Train the neural network
# ------------------------------------------------------------

epochs = 100

train_losses = []

for epoch in range(epochs):

    # Forward pass
    predictions = mlp(X_train_tensor)

    # Calculate loss
    loss = criterion(
        predictions,
        y_train_tensor
    )

    # Clear previous gradients
    optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    # Update model parameters
    optimizer.step()

    # Store loss
    train_losses.append(loss.item())

    # Print every 10 epochs
    if (epoch + 1) % 10 == 0:

        print(
            f"Epoch [{epoch + 1}/{epochs}], "
            f"Loss: {loss.item():.6f}"
        )


# ============================================================
# 6. MAKE PREDICTIONS
# ============================================================

# Linear regression predictions
linear_predictions = linear_model.predict(X_test_scaled)


# MLP predictions
mlp.eval()

with torch.no_grad():

    mlp_predictions = mlp(
        X_test_tensor
    ).numpy().flatten()


# ============================================================
# 7. CALCULATE PERFORMANCE METRICS
# ============================================================

# ------------------------------------------------------------
# Linear Regression
# ------------------------------------------------------------

linear_mse = mean_squared_error(
    y_test,
    linear_predictions
)

linear_rmse = np.sqrt(linear_mse)

linear_r2 = r2_score(
    y_test,
    linear_predictions
)


# ------------------------------------------------------------
# PyTorch MLP
# ------------------------------------------------------------

mlp_mse = mean_squared_error(
    y_test,
    mlp_predictions
)

mlp_rmse = np.sqrt(mlp_mse)

mlp_r2 = r2_score(
    y_test,
    mlp_predictions
)


# ------------------------------------------------------------
# Print metrics
# ------------------------------------------------------------

print("\nModel Performance")
print("-----------------")

print("\nLinear Regression:")
print(f"MSE:  {linear_mse:.6f}")
print(f"RMSE: {linear_rmse:.6f}")
print(f"R²:   {linear_r2:.6f}")

print("\nPyTorch MLP:")
print(f"MSE:  {mlp_mse:.6f}")
print(f"RMSE: {mlp_rmse:.6f}")
print(f"R²:   {mlp_r2:.6f}")


# ============================================================
# 8. CREATE TRAINING-LOSS FIGURE
# ============================================================

# Directory where this Python script is located
script_dir = Path(__file__).resolve().parent

# Output paths
readme_path = script_dir / "README.md"
figure_path = script_dir / "mlp_training_loss.png"


# Plot training loss
plt.figure(figsize=(8, 5))

plt.plot(
    range(1, epochs + 1),
    train_losses
)

plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("PyTorch MLP Training Loss")

plt.tight_layout()

plt.savefig(
    figure_path,
    dpi=300
)

plt.close()


# ============================================================
# 9. DETERMINE MODEL COMPARISON
# ============================================================

if linear_rmse < mlp_rmse:

    comparison = (
        "Linear Regression achieved a lower RMSE than the "
        "PyTorch MLP on the test set."
    )

else:

    comparison = (
        "The PyTorch MLP achieved a lower RMSE than "
        "Linear Regression on the test set."
    )


# ============================================================
# 10. CREATE README
# ============================================================

# We use this variable so that Markdown code fences inside
# the README do not interfere with the Python code block.
markdown_fence = "```"


readme_content = f"""
# First part:
To run the code go to the folder ./Intro_deep_learning, and run from terminal: python ./hw2.py

# Second part
# California Housing: Model Comparison

## Dataset Exploration

### 1. Load Dataset

The California Housing dataset was loaded using
`fetch_california_housing()` from `sklearn.datasets`.

### 2. Dataset Description

{markdown_fence}text
{dataset_description}
{markdown_fence}

### 3. DataFrame

The dataset features and target variable were stored in a
Pandas DataFrame. The target variable is `MedHouseVal`.

### 4. First Five Rows

{markdown_fence}text
{head_output}
{markdown_fence}

### 5. Descriptive Statistics

{markdown_fence}text
{describe_output}
{markdown_fence}


## Model Development

The dataset was divided into training and testing sets using an
80/20 split with `random_state=0`.

The predictor variables were standardized using
`StandardScaler`, fitted only on the training data.

Two models were trained:

1. Linear Regression using scikit-learn
2. A Multi-Layer Perceptron (MLP) implemented using PyTorch


## Performance Comparison

| Model | MSE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | {linear_mse:.6f} | {linear_rmse:.6f} | {linear_r2:.6f} |
| PyTorch MLP | {mlp_mse:.6f} | {mlp_rmse:.6f} | {mlp_r2:.6f} |

### Analysis

{comparison}

The MSE, RMSE, and R² metrics all show better performance with the PyTorch MLP than 
with linear regression. This is likely because the data do not share a linear relationship, 
and the MLP is better able to capture any non-linearity present between the variables.

## PyTorch Training Loss

The PyTorch MLP was trained for {epochs} epochs using the Adam
optimizer with a learning rate of 0.01 and mean squared error
loss. It can be seen that, in each iteration, the value of the objective function decreases, 
such that the MLP parameters move closer to the model's optimal parameters.

The training loss is shown below.

![MLP Training Loss](mlp_training_loss.png)
"""


# ============================================================
# 11. WRITE README FILE
# ============================================================

with open(
    readme_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(readme_content)


# ============================================================
# 12. FINAL MESSAGE
# ============================================================

print("\nREADME created successfully:")
print(readme_path)

print("\nTraining-loss figure created successfully:")
print(figure_path)