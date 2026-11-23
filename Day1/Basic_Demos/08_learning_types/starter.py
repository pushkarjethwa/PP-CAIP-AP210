"""Day 1 | Slides 8 | Supervised, unsupervised, reinforcement, and other learning types (starter)."""

# %% Setup

from collections import Counter
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.semi_supervised import LabelSpreading

x = [[0], [0.2], [0.4], [5], [5.2], [5.4]]

# Fill each TODO in order. Run the file after you add each step.

# An untouched starter only loads the setup and lists the steps.

# %% 1. Supervised

# TODO Fit a linear regression model on [[1], [2], [4]] with labels [2, 4, 8].
print("\nStep 1: Supervised")

# %% 2. Unsupervised clustering

# TODO Fit KMeans with two clusters and print its label for every row.
print("\nStep 2: Unsupervised clustering")

# %% 3. Unsupervised dimension reduction

# TODO Run PCA with n_components=1 on three two-dimensional points and print the
#      shape.
print("\nStep 3: Unsupervised dimension reduction")

# %% 4. Unsupervised anomaly detection

# TODO Fit an IsolationForest with contamination=0.2 on five points including 20.
print("\nStep 4: Unsupervised anomaly detection")

# %% 5. Reinforcement learning

# TODO For each (action, reward) pair, update its value halfway towards the
#      reward.
print("\nStep 5: Reinforcement learning")

# %% 6. Semi-supervised

# TODO Fit LabelSpreading on x with labels [0, -1, -1, -1, -1, 1].
print("\nStep 6: Semi-supervised")

# %% 7. Self-supervised

# TODO Make adjacent token pairs, then count the tokens that follow A.
print("\nStep 7: Self-supervised")
