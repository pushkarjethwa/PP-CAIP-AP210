"""Day 1 | Slides 19 | One-hot, ordinal, rare categories, and target encoding (starter)."""

# %% Setup

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder

colors = [["red"], ["red"], ["blue"], ["blue"], ["green"], ["yellow"]]
sizes = [["small"], ["medium"], ["large"]]
ids = np.array([["a"], ["b"], ["c"], ["d"], ["e"], ["f"], ["g"], ["h"]])
target = np.array([0., 1., 0., 1., 1., 0., 1., 0.])

# Fill each TODO in order. Run the file after you add each step.

# An untouched starter only loads the setup and lists the steps.

# %% 1. One-hot encoding

# TODO Fit OneHotEncoder with dense output and print feature names and encoded
#      rows.
print("\nStep 1: One-hot encoding")

# %% 2. Unknown category

# TODO Transform [['purple']] with onehot and print the result.
print("\nStep 2: Unknown category")

# %% 3. Ordinal encoding

# TODO Fit OrdinalEncoder with categories specified in size order.
print("\nStep 3: Ordinal encoding")

# %% 4. Group rare categories

# TODO Fit OneHotEncoder with min_frequency=2 and
#      handle_unknown='infrequent_if_exist'.
print("\nStep 4: Group rare categories")

# %% 5. Target encoding with cross-fitting

# TODO Compare the same-row group mean to TargetEncoder.fit_transform and
#      transform an unseen ID.
print("\nStep 5: Target encoding with cross-fitting")
