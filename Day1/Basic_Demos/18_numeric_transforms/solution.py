"""Day 1 | Slides 18 | Scaling, log, Box-Cox, and binning (complete solution)."""

# %% Setup

import sys

import numpy as np
import pandas as pd
from scipy.stats import boxcox, skew
from sklearn.preprocessing import MinMaxScaler, StandardScaler

train = np.array([1, 2, 4, 8, 32, 128], dtype=float).reshape(-1, 1)

# %% 1. Standardisation

print("\nStep 1: Standardisation")
scaler = StandardScaler().fit(train)
standard = scaler.transform(train).ravel()
print("Standardised:", standard.round(2))
print(f"Mean {standard.mean():.2f} | Population SD {standard.std():.2f}")

# %% 2. Min-max and outlier sensitivity

print("\nStep 2: Min-max and outlier sensitivity")
minmax = MinMaxScaler().fit_transform(train).ravel()
without_outlier = MinMaxScaler().fit_transform(train[:-1]).ravel()
print("Min-max with 128:", minmax.round(2))
print("Min-max without 128:", without_outlier.round(2))

# %% 3. Log and Box-Cox

print("\nStep 3: Log and Box-Cox")
logged = np.log(train.ravel())
powered, power = boxcox(train.ravel())
for name, values in [("Raw", train.ravel()), ("Standard", standard),
                     ("Min-max", minmax), ("Log", logged), ("Box-Cox", powered)]:
    print(name, "skew:", round(skew(values), 2))
print("Fitted Box-Cox power:", round(power, 2))

# %% 4. Binning

print("\nStep 4: Binning")
bands = pd.cut(train.ravel(), bins=[0, 5, 35, np.inf],
               labels=["small", "medium", "large"], right=False)
print("Bands:", list(zip(train.ravel().tolist(), bands.tolist())))

# %% 5. Fit on training, reuse on new values

print("\nStep 5: Fit on training, reuse on new values")
print("New min-max value:", MinMaxScaler().fit(train).transform([[256]]).round(2))
print("New standard value:", scaler.transform([[256]]).round(2))
print("Unchanged training mean:", scaler.mean_)

# %% 6. See the shape

print("\nStep 6: See the shape")
if "--plot" in sys.argv:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(10, 3))
    for ax, (name, values) in zip(axes, [("Raw", train.ravel()),
             ("Standard", standard), ("Min-max", minmax), ("Log", logged)]):
        ax.hist(values, bins=5)
        ax.set(title=name, xlabel="Value", ylabel="Count")
    fig.tight_layout()
    plt.show()
