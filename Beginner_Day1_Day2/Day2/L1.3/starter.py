""" L1.3 - Transaction features

We will turn several event rows into one row per item, then check what the new values mean. This makes event data easier to use in a model.
"""

# %% Setup

import numpy as np
import pandas as pd

# %% 1. Load and compare memory

# Categories can reduce memory when text values repeat.

print("\nStep 1 - Load and compare memory")

events = pd.DataFrame({'id': ['a','a','b','b'], 'type': ['x','x','y','y'], 'amount': [10,20,30,40]})
print(events)
print('Text bytes:', events['type'].memory_usage(deep=True))
print('Category bytes:', events['type'].astype('category').memory_usage(deep=True))

# %% 2. Aggregate event rows

# We calculate count and total for each ID.

print("\nStep 2 - Aggregate event rows")

events = pd.DataFrame({'id': ['a','a','b'], 'amount': [10,20,30]})
print(events.groupby('id')['amount'].agg(['count','sum']))

# %% 3. Spend mix and trend

# We compare each category's share and the totals in two periods.

print("\nStep 3 - Spend mix and trend")

events = pd.DataFrame({'type': ['x','y','x'], 'period': ['early','early','late'], 'amount': [10,30,20]})
print('Shares:', (events.groupby('type')['amount'].sum() / events['amount'].sum()).round(2))
print('Period totals:', events.groupby('period')['amount'].sum())

# %% 4. Join and handle no activity

# An item with no events needs a flag; a missing average is not a measured zero.

print("\nStep 4 - Join and handle no activity")

items = pd.DataFrame({'id': ['a','b','c']})
totals = pd.DataFrame({'id': ['a','b'], 'total': [30,40]})
joined = items.merge(totals, on='id', how='left')
joined['has_activity'] = joined['total'].notna()
joined['total'] = joined['total'].fillna(0)
print(joined)

# %% 5. Compare a feature with random noise

# We compare how much a signal and random values relate to a target.

print("\nStep 5 - Compare a feature with random noise")

rng = np.random.default_rng(2)
target = pd.Series([1,2,3,4,5,6])
signal = pd.Series([1,2,3,4,5,6])
noise = pd.Series(rng.normal(size=6))
print('Signal correlation:', signal.corr(target))
print('Noise correlation:', round(noise.corr(target), 2))

# %% 6. Turn an image into numbers

# An image can be stored as a grid of pixel values.

print("\nStep 6 - Turn an image into numbers")

image = np.array([[0,255],[255,0]])
print('Image:', image)
print('Pixels:', image.ravel())

# %% 7. Describe a data contract

# We record column names and meanings so another step knows what it receives.

print("\nStep 7 - Describe a data contract")

contract = {'id': 'unique identifier', 'total': 'sum of activity', 'has_activity': 'whether events exist'}
for column, meaning in contract.items():
    print(column, '-', meaning)

# PRACTISE: Change one numeric input in one step and rerun that step.
# Compare with the original output. Restore the value and try another step.
