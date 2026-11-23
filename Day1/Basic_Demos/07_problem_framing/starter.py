"""Day 1 | Slides 7 | Observation, features, target, horizon, and metric (starter)."""

# %% Setup

import pandas as pd
from sklearn.metrics import mean_absolute_error

series = pd.Series([10, 13, 12, 16, 15], name="reading")

# Fill each TODO in order. Run the file after you add each step.

# An untouched starter only loads the setup and lists the steps.

# %% 1. Build one row per time point

# TODO Convert the Series to a DataFrame and add a one-step lag using shift(1).
print("\nStep 1: Build one row per time point")

# %% 2. Define a one-step target

# TODO Add the next reading with shift(-1), then print the history with labels.
print("\nStep 2: Define a one-step target")

# %% 3. Score a baseline

# TODO Use mean_absolute_error on next_reading versus reading in labelled.
print("\nStep 3: Score a baseline")
