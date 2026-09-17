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

# 1.1 Load the dataset
housing = fetch_california_housing()

# 1.2 Print the dataset description
print(housing.DESCR)

# 1.3 Create a DataFrame from the feature data
df = pd.DataFrame(
    housing.data,
    columns=housing.feature_names
)

# Add the target variable
df["MedHouseVal"] = housing.target

# 1.4 Display the first 5 rows
print("\nFirst 5 rows:")
print(df.head())

# 1.5 Display summary statistics
print("\nSummary statistics:")
print(df.describe())

# 2.1 Separate features (X) and target (y)
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=0
)

# 2.2 Initialize the scaler
scaler = StandardScaler()

# Fit the scaler ONLY on the training data
X_train_scaled = scaler.fit_transform(X_train)

# Use the same scaler to transform the testing data
X_test_scaled = scaler.transform(X_test)

# 3.1 Build: Create the Linear Regression model
lr_model = LinearRegression()

# Train: Fit the model using scaled training data
lr_model.fit(X_train_scaled, y_train)

print("Linear Regression model trained successfully.")

# 3.2.A Convert scaled training and testing features to tensors
X_train_tensor = torch.tensor(
    X_train_scaled, dtype=torch.float32
)

X_test_tensor = torch.tensor(
    X_test_scaled, dtype=torch.float32
)

# Convert target variables to tensors
# Reshape targets to (n_samples, 1)
y_train_tensor = torch.tensor(
    y_train.to_numpy(), dtype=torch.float32
).view(-1, 1)

y_test_tensor = torch.tensor(
    y_test.to_numpy(), dtype=torch.float32
).view(-1, 1)

# 3.2.B Define the shallow neural network
class MLP(nn.Module):

    def __init__(self):
        super().__init__()

        # Input layer: 8 features -> 32 neurons
        self.hidden = nn.Linear(8, 32)

        # Output layer: 32 neurons -> 1 output
        self.output = nn.Linear(32, 1)

    def forward(self, x):

        # Hidden layer followed by ReLU activation
        x = torch.relu(self.hidden(x))

        # Output layer
        x = self.output(x)

        return x

# 3.2.C Create the neural network
mlp_model = MLP()

# Loss function for regression
criterion = nn.MSELoss()

# Adam optimizer with learning rate 0.01
optimizer = optim.Adam(
    mlp_model.parameters(),
    lr=0.01
)

# 3.2.D Number of training epochs
num_epochs = 100

# Store the loss for every epoch
train_losses = []

for epoch in range(num_epochs):

    # Forward pass
    predictions = mlp_model(X_train_tensor)

    # Calculate loss
    loss = criterion(predictions, y_train_tensor)

    # Backward pass and weight update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Store the loss
    train_losses.append(loss.item())

    # Print loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Loss: {loss.item():.4f}"
        )

# 4.1 Make predictions using Linear Regression

# Predictions on the scaled test set
y_pred_lr = lr_model.predict(X_test_scaled)

# Make predictions using the PyTorch Neural Network

# Disable gradient calculations during evaluation
with torch.no_grad():
    y_pred_mlp = mlp_model(X_test_tensor)

# Convert PyTorch tensor predictions to NumPy array
y_pred_mlp = y_pred_mlp.numpy().flatten()

# Convert test target to NumPy array
y_test_numpy = y_test.to_numpy()

# 4.2 Calculate metrics for Linear Regression

mse_lr = mean_squared_error(y_test_numpy, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
r2_lr = r2_score(y_test_numpy, y_pred_lr)

# Calculate metrics for PyTorch Neural Network

mse_mlp = mean_squared_error(y_test_numpy, y_pred_mlp)
rmse_mlp = np.sqrt(mse_mlp)
r2_mlp = r2_score(y_test_numpy, y_pred_mlp)

# 5.1 Plot Training Loss

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, num_epochs + 1),
    train_losses,
    label="Training Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("PyTorch MLP Training Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()

# Save the figure in the same directory as the script
plt.savefig("mlp_training_loss.png", dpi=300)

plt.show()

# 5.2 Update README.md

# Use the directory where this Python script is located
script_dir = Path(__file__).resolve().parent

# File paths
readme_path = script_dir / "README.md"
figure_path = script_dir / "mlp_training_loss.png"

# Determine which model performed better

# Lower RMSE means smaller prediction errors
if rmse_mlp < rmse_lr:
    better_model = "PyTorch Neural Network"
    comparison = (
        "The PyTorch Neural Network performed better than "
        "Linear Regression on the test data because it achieved "
        "a lower RMSE."
    )
elif rmse_lr < rmse_mlp:
    better_model = "Linear Regression"
    comparison = (
        "Linear Regression performed better than the PyTorch "
        "Neural Network on the test data because it achieved "
        "a lower RMSE."
    )
else:
    better_model = "Both models"
    comparison = (
        "Both models achieved the same RMSE on the test data."
    )

# ------------------------------------------------------------
# Create the README content
# ------------------------------------------------------------

readme_content = f"""# California Housing: Model Comparison

## Performance Comparison

The following table compares the performance of the baseline
Linear Regression model and the PyTorch Neural Network.

| Model | MSE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | {mse_lr:.4f} | {rmse_lr:.4f} | {r2_lr:.4f} |
| PyTorch Neural Network | {mse_mlp:.4f} | {rmse_mlp:.4f} | {r2_mlp:.4f} |

### Analysis

{comparison}

The MSE measures the average squared prediction error, while
the RMSE expresses the prediction error in the same units as
the target variable. The R² value measures the proportion of
variance in the target explained by the model.

The target variable, MedHouseVal, is expressed in hundreds of
thousands of dollars. Therefore, the RMSE can be interpreted
in those same units.

## PyTorch Neural Network Training Loss

The following figure shows the training loss over 100 epochs.

![PyTorch MLP Training Loss](mlp_training_loss.png)

### Training Progress

The neural network's training loss is tracked at every epoch.
As the number of epochs increases, the optimizer updates the
network weights to reduce the Mean Squared Error.

A decreasing loss indicates that the model is learning to
reduce its prediction errors on the training data. If the loss
decreases rapidly during the initial epochs and then levels
off, the model is approaching a stable training solution.

The final loss should be examined together with the test-set
metrics to determine whether the model generalizes well to
unseen data.
"""

# Write the README file
readme_path.write_text(readme_content, encoding="utf-8")

print(f"README updated: {readme_path}")
print(f"Figure saved: {figure_path}")