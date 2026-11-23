"""Day 1 | Slides 17 | Why values go missing and how to handle them (starter)."""

# %% Setup

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(42)
full = pd.DataFrame({"group": ["a", "a", "b", "b", "a", "b"],
                     "value": [2, 4, 6, 8, 10, 12]})
incomplete = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6],
                           "group": ["a", "a", "a", "b", "b", "b"],
                           "value": [2, np.nan, 6, 8, np.nan, 12]})

# Fill each TODO in order. Run the file after you add each step.

# An untouched starter only loads the setup and lists the steps.

# %% 1. MCAR

# TODO Use rng.choice to select two row indices and blank the corresponding
#      values.
print("\nStep 1: MCAR")

# %% 2. MAR

# TODO Mask values for rows where group is b and print the result.
print("\nStep 2: MAR")

# %% 3. MNAR

# TODO Mask values above 8 and print the resulting Series.
print("\nStep 3: MNAR")

# %% 4. Drop rows or fill a simple median

# TODO Drop rows missing value and compare with filling from the overall median.
print("\nStep 4: Drop rows or fill a simple median")

# %% 5. Fill using groups and keep an indicator

# TODO Use groupby(...).transform('median') to fill values and save the original
#      missingness flag.
print("\nStep 5: Fill using groups and keep an indicator")

# %% 6. Model-based fill

# TODO Fit LinearRegression on rows with known values and predict for missing
#      rows.
print("\nStep 6: Model-based fill")
