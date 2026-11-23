"""Day 1 | Slides 11 | Python, NumPy, pandas, and the scikit-learn API (starter)."""

# %% Setup

from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

values = np.array([10, 20, 30])
big = np.arange(300_000)
table = pd.DataFrame({"id": [1, 2, 3, 4], "group": ["a", "a", "b", "b"],
                      "value": [2, 4, 6, 8]})
labels = pd.DataFrame({"group": ["a", "b"], "name": ["alpha", "beta"]})
train_x = [[1], [2], [3]]
train_y = [2, 4, 6]

# Fill each TODO in order. Run the file after you add each step.

# An untouched starter only loads the setup and lists the steps.

# %% 1. Array, shape, dtype, and boolean mask

# TODO Print values.shape and values.dtype, then mask for values >= 20.
print("\nStep 1: Array, shape, dtype, and boolean mask")

# %% 2. Vectorized maths vs a Python loop

# TODO Use perf_counter on a loop that doubles big and an array expression that
#      does the same.
print("\nStep 2: Vectorized maths vs a Python loop")

# %% 3. DataFrame, Series, dtypes, and apply

# TODO Print dtypes, select the value Series, and compare apply with value * 2.
print("\nStep 3: DataFrame, Series, dtypes, and apply")

# %% 4. Chaining and named aggregation

# TODO Add high where value > 4, group by group and high, and calculate count and
#      mean.
print("\nStep 4: Chaining and named aggregation")

# %% 5. Merge tables

# TODO Left-merge table and labels on group, then print the result.
print("\nStep 5: Merge tables")

# %% 6. Pure transformation

# TODO Define a function returning frame.assign(doubled=frame['value'] * 2).
print("\nStep 6: Pure transformation")

# %% 7. fit, transform, predict, and Pipeline

# TODO Fit a scaler, transform [[4]], then fit a scaling-and-regression Pipeline
#      and predict [[4]].
print("\nStep 7: fit, transform, predict, and Pipeline")
