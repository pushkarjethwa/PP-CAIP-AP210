"""Day 1 | Slides 14 | Vectors, dot products, loss, gradients, and learning rate (starter)."""

# %% Setup

import sys

import numpy as np
from sklearn.metrics import log_loss

x = np.array([2, 3])
w = np.array([4, 1])
b = 1

# Fill each TODO in order. Run the file after you add each step.

# An untouched starter only loads the setup and lists the steps.

# %% 1. Vector and dot product

# TODO Calculate x @ w, then add b to get the prediction.
print("\nStep 1: Vector and dot product")

# %% 2. Mean squared error

# TODO Compute squared errors for predictions [2, 7] and actual values [1, 5].
print("\nStep 2: Mean squared error")

# %% 3. Binary cross-entropy

# TODO Compare log_loss for [0.1, 0.9] and [0.9, 0.1] when the labels are [0, 1].
print("\nStep 3: Binary cross-entropy")

# %% 4. Gradient descent and learning rate

# TODO Run four updates for the loss (weight - 3) squared, with rates 0.25 and
#      1.1.
print("\nStep 4: Gradient descent and learning rate")

# %% 5. Show the loss curve

# TODO If --plot is present, plot the loss against weight and overlay paths[0.25].
print("\nStep 5: Show the loss curve")
