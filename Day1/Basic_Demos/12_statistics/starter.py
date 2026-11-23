"""Day 1 | Slides 12 | Centre, spread, skew, and four distributions (starter)."""

# %% Setup

import sys

import numpy as np
import pandas as pd

values = pd.Series([1, 2, 2, 3, 12])
rng = np.random.default_rng(42)

# Fill each TODO in order. Run the file after you add each step.

# An untouched starter only loads the setup and lists the steps.

# %% 1. Mean and median

# TODO Print the mean and median of values.
print("\nStep 1: Mean and median")

# %% 2. Variance and standard deviation

# TODO Print var() and std() on the Series, rounded to two decimals.
print("\nStep 2: Variance and standard deviation")

# %% 3. Skew and a log

# TODO Compare values.skew() with np.log(values).skew().
print("\nStep 3: Skew and a log")

# %% 4. Normal, log-normal, binomial, and Poisson

# TODO Generate four samples with rng and plot a histogram for each if --plot is
#      present.
print("\nStep 4: Normal, log-normal, binomial, and Poisson")
