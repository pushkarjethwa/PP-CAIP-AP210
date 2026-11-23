"""Day 1 | Slides 11 | Python, NumPy, pandas, and the scikit-learn API (complete solution)."""

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

# %% 1. Array, shape, dtype, and boolean mask

print("\nStep 1: Array, shape, dtype, and boolean mask")
print("Shape:", values.shape, "dtype:", values.dtype)
mask = values >= 20
print("Mask:", mask.tolist(), "Selected:", values[mask].tolist())

# %% 2. Vectorized maths vs a Python loop

print("\nStep 2: Vectorized maths vs a Python loop")
start = perf_counter()
doubled_loop = []
for value in big:
    doubled_loop.append(value * 2)
loop_seconds = perf_counter() - start
start = perf_counter()
doubled_array = big * 2
array_seconds = perf_counter() - start
print("Same values:", np.array_equal(doubled_loop, doubled_array))
print(f"Loop: {loop_seconds:.5f}s | Array: {array_seconds:.5f}s")

# %% 3. DataFrame, Series, dtypes, and apply

print("\nStep 3: DataFrame, Series, dtypes, and apply")
column = table["value"]
print(type(table).__name__, type(column).__name__)
print(table.dtypes)
print("Same result:", column.apply(lambda x: x * 2).equals(column * 2))

# %% 4. Chaining and named aggregation

print("\nStep 4: Chaining and named aggregation")
summary = (table.assign(high=table["value"] > 4)
           .groupby(["group", "high"])
           .agg(count=("id", "size"), mean=("value", "mean")))
print(summary)

# %% 5. Merge tables

print("\nStep 5: Merge tables")
print(table.merge(labels, on="group", how="left"))

# %% 6. Pure transformation

print("\nStep 6: Pure transformation")
def add_doubled(frame):
    return frame.assign(doubled=frame["value"] * 2)

print(add_doubled(table))
print("Original columns:", table.columns.tolist())

# %% 7. fit, transform, predict, and Pipeline

print("\nStep 7: fit, transform, predict, and Pipeline")
scaler = StandardScaler().fit(train_x)
print("Learned mean:", scaler.mean_, "Scaled 4:", scaler.transform([[4]]))
pipe = Pipeline([("scale", StandardScaler()), ("model", LinearRegression())])
pipe.fit(train_x, train_y)
print("Prediction for 4:", pipe.predict([[4]]).round(2))
