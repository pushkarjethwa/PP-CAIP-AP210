"""Day 1 | Slides 7 | Observation, features, target, horizon, and metric (complete solution)."""

# %% Setup

import pandas as pd
from sklearn.metrics import mean_absolute_error

series = pd.Series([10, 13, 12, 16, 15], name="reading")

# %% 1. Build one row per time point

print("\nStep 1: Build one row per time point")
rows = series.to_frame().assign(previous=series.shift(1))
print(rows)

# %% 2. Define a one-step target

print("\nStep 2: Define a one-step target")
rows["next_reading"] = series.shift(-1)
print(rows)
labelled = rows.dropna(subset=["previous", "next_reading"])

# %% 3. Score a baseline

print("\nStep 3: Score a baseline")
error = mean_absolute_error(labelled["next_reading"], labelled["reading"])
print("Baseline MAE:", error)
