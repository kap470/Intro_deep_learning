# California Housing: Model Comparison

## Performance Comparison

The following table compares the performance of the baseline
Linear Regression model and the PyTorch Neural Network.

| Model | MSE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 0.5290 | 0.7273 | 0.5943 |
| PyTorch Neural Network | 0.4469 | 0.6685 | 0.6573 |

### Analysis

The PyTorch Neural Network performed better than Linear Regression on the test data because it achieved a lower RMSE.

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
