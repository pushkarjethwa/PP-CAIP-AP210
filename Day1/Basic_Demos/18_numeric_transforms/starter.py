"""Day 1 | Slides 18 | Scaling, log, Box-Cox, and binning (starter)."""

# %% Setup

import sys

import numpy as np
import pandas as pd
from scipy.stats import boxcox, skew
from sklearn.preprocessing import MinMaxScaler, StandardScaler

train = np.array([1, 2, 4, 8, 32, 128], dtype=float).reshape(-1, 1)

# Fill each TODO in order. Run the file after you add each step.

# An untouched starter only loads the setup and lists the steps.

# %% 1. Standardisation

# TODO Fit StandardScaler on train and print the transformed values, their mean,
#      and standard deviation.
print("\nStep 1: Standardisation")

# %% 2. Min-max and outlier sensitivity

# TODO Fit MinMaxScaler with and without the final outlier and compare the shared
#      values.
print("\nStep 2: Min-max and outlier sensitivity")

# %% 3. Log and Box-Cox

# TODO Compare skew after np.log and scipy.stats.boxcox with skew before
#      transformation.
print("\nStep 3: Log and Box-Cox")

# %% 4. Binning

# TODO Use pd.cut with cut points 0, 5, 35, and infinity and three band names.
print("\nStep 4: Binning")

# %% 5. Fit on training, reuse on new values

# TODO Transform [[256]] using min-max fitted on train and the already-fitted
#      standard scaler.
print("\nStep 5: Fit on training, reuse on new values")

# %% 6. See the shape

# TODO If --plot is present, create four histograms of raw, standard, min-max, and
#      log values.
print("\nStep 6: See the shape")
