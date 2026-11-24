""" D2.2 - Time-series patterns

We will arrange observations in time and inspect their pattern. We will build features from earlier values and keep future rows for checking.
"""

# %% Setup

import numpy as np
import pandas as pd

# %% 1. Plot and inspect a sequence

# Time order helps us see trend and repeated patterns.

print("\nStep 1 - Plot and inspect a sequence")

series = pd.Series([10,12,11,13,12,14,13,15])
print(series)
print('First:', series.iloc[0], 'Last:', series.iloc[-1])

# %% 2. Trend and repeated positions

# A rolling mean shows local level. Grouping by position can reveal a repeating cycle.

print("\nStep 2 - Trend and repeated positions")

series = pd.Series([10,12,11,13,12,14,13,15])
print('Rolling mean:', series.rolling(3).mean().round(1).tolist())
print('Position averages:', series.groupby(series.index % 2).mean().tolist())

# %% 3. Autocorrelation

# We compare the series with its previous value.

print("\nStep 3 - Autocorrelation")

series = pd.Series([10,12,11,13,12,14,13,15])
print('Lag 1 correlation:', round(series.autocorr(lag=1),2))

# %% 4. Lag and rolling average without leakage

# We shift first so a feature only uses earlier values.

print("\nStep 4 - Lag and rolling average without leakage")

series = pd.Series([10,12,11,13,12])
table = pd.DataFrame({'value': series})
table['lag1'] = series.shift(1)
table['past_mean'] = series.shift(1).rolling(2).mean()
print(table)

# %% 5. Ordered splits

# Each training set ends before its test set begins.

print("\nStep 5 - Ordered splits")

from sklearn.model_selection import TimeSeriesSplit
for train, test in TimeSeriesSplit(n_splits=3).split(np.arange(8)):
    print('Train:', train, 'Test:', test)
