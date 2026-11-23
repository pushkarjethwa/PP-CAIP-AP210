"""Day 1 | Slides 5 | AI, ML, deep learning, and data science (starter)."""

# %% Setup

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier

numbers = [1, 2, 3, 4]
x = [[1], [2], [3], [4]]
y = [3, 5, 7, 9]
bits = [[0, 0], [0, 1], [1, 0], [1, 1]]
different = [0, 1, 1, 0]

# Fill each TODO in order. Run the file after you add each step.

# An untouched starter only loads the setup and lists the steps.

# %% 1. Written rule

# TODO Apply value >= 3 to every number and print the resulting booleans.
print("\nStep 1: Written rule")

# %% 2. Learned rule

# TODO Fit LinearRegression to x and y, then predict for [[5]].
print("\nStep 2: Learned rule")

# %% 3. Neural network

# TODO Fit an MLPClassifier with two hidden layers and print its predictions.
print("\nStep 3: Neural network")

# %% 4. Data science without a model

# TODO Make a pandas Series from numbers and print its count and mean.
print("\nStep 4: Data science without a model")
