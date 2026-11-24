""" L2.2 - Forecasting time series

We will fill a short gap, make a simple forecast and examine methods that use trends, calendars and uncertainty.
"""

# %% Setup

import numpy as np
import pandas as pd

# %% 1. Find and fill an outage gap

# We fill the gap using a past value and retain a flag.

print("\nStep 1 - Find and fill an outage gap")

series = pd.Series([10,12,None,13,14])
missing = series.isna()
print(pd.DataFrame({'original':series, 'filled':series.ffill(), 'was_missing':missing}))

# %% 2. Compare simple baselines

# We compare yesterday's value and the average of earlier observations.

print("\nStep 2 - Compare simple baselines")

history = pd.Series([10,12,11,13])
actual = 14
print('Last value:', history.iloc[-1], 'error:', abs(actual-history.iloc[-1]))
print('Mean:', history.mean(), 'error:', abs(actual-history.mean()))

# %% 3. Trend, seasonality and calendar inputs

# We inspect a repeating sequence and a calendar flag. Forecasting models can use these patterns.

print("\nStep 3 - Trend, seasonality and calendar inputs")

values = pd.DataFrame({'day':[1,2,3,4], 'weekend':[0,1,0,1], 'value':[10,15,11,16]})
print(values.groupby('weekend')['value'].mean())

# %% 4. Build lagged features

# Previous observations can help predict a later observation.

print("\nStep 4 - Build lagged features")

values = pd.Series([10,12,11,13,14])
print(pd.DataFrame({'value':values, 'previous':values.shift(1), 'past_average':values.shift(1).rolling(2).mean()}))

# %% 5. Compare models with ordered splits

# TimeSeriesSplit holds later rows out on each round.

print("\nStep 5 - Compare models with ordered splits")

from sklearn.model_selection import TimeSeriesSplit
for train, test in TimeSeriesSplit(n_splits=2).split(np.arange(6)):
    print('Train:',train,'Test:',test)

# %% 6. Quantile forecast

# A 90th percentile represents a high value in the observed sample.

print("\nStep 6 - Quantile forecast")

observations = pd.Series([10,11,12,13,14,15,16,17,18,19])
print('Sample 90th percentile:', round(observations.quantile(0.9),1))
