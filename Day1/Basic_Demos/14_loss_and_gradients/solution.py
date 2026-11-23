"""Day 1 | Slides 14 | Vectors, dot products, loss, gradients, and learning rate (complete solution)."""

# %% Setup

import sys

import numpy as np
from sklearn.metrics import log_loss

x = np.array([2, 3])
w = np.array([4, 1])
b = 1

# %% 1. Vector and dot product

print("\nStep 1: Vector and dot product")
score = x @ w
print("Dot product:", score, "Prediction:", score + b)

# %% 2. Mean squared error

print("\nStep 2: Mean squared error")
squared_errors = (np.array([2, 7]) - np.array([1, 5])) ** 2
print("Squared errors:", squared_errors.tolist(), "MSE:", squared_errors.mean())

# %% 3. Binary cross-entropy

print("\nStep 3: Binary cross-entropy")
print("Well matched:", round(log_loss([0, 1], [0.1, 0.9]), 3))
print("Confident errors:", round(log_loss([0, 1], [0.9, 0.1]), 3))

# %% 4. Gradient descent and learning rate

print("\nStep 4: Gradient descent and learning rate")
paths = {}
for rate in [0.25, 1.1]:
    weight = 0.0
    weights = [weight]
    for _ in range(4):
        gradient = 2 * (weight - 3)
        weight -= rate * gradient
        weights.append(weight)
    paths[rate] = np.array(weights)
    print("Rate:", rate, "Weights:", paths[rate].round(2),
          "Losses:", ((paths[rate] - 3) ** 2).round(2))

# %% 5. Show the loss curve

print("\nStep 5: Show the loss curve")
if "--plot" in sys.argv:
    import matplotlib.pyplot as plt

    grid = np.linspace(-0.5, 6.5, 150)
    plt.plot(grid, (grid - 3) ** 2, label="Loss")
    plt.scatter(paths[0.25], (paths[0.25] - 3) ** 2, label="Descent steps")
    plt.xlabel("Weight")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()
