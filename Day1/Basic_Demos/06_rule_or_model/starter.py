"""Day 1 | Slides 6 | When a rule or a model makes sense (starter)."""

# %% Setup

from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier

values = [2, 4, 6]
labels_by_code = {"A": "alpha", "B": "beta"}
x_train = [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]
y_train = [0, 0, 1, 1, 1, 1]
x_check = [[0, 0], [1, 1], [2, 1]]
y_check = [0, 1, 0]

# Fill each TODO in order. Run the file after you add each step.

# An untouched starter only loads the setup and lists the steps.

# %% 1. Fixed rule and lookup

# TODO Check whether each value exceeds 4, and look up the name for code B.
print("\nStep 1: Fixed rule and lookup")

# %% 2. Learn and check mistakes

# TODO Fit a shallow tree, predict the check rows, and print the confusion matrix.
print("\nStep 2: Learn and check mistakes")
