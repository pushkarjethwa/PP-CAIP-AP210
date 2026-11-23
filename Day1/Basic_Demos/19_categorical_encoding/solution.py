"""Day 1 | Slides 19 | One-hot, ordinal, rare categories, and target encoding (complete solution)."""

# %% Setup

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder

colors = [["red"], ["red"], ["blue"], ["blue"], ["green"], ["yellow"]]
sizes = [["small"], ["medium"], ["large"]]
ids = np.array([["a"], ["b"], ["c"], ["d"], ["e"], ["f"], ["g"], ["h"]])
target = np.array([0., 1., 0., 1., 1., 0., 1., 0.])

# %% 1. One-hot encoding

print("\nStep 1: One-hot encoding")
onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(colors)
print(onehot.get_feature_names_out().tolist())
print(onehot.transform([["red"], ["blue"]]).astype(int))

# %% 2. Unknown category

print("\nStep 2: Unknown category")
print("Unseen purple:", onehot.transform([["purple"]]).astype(int))

# %% 3. Ordinal encoding

print("\nStep 3: Ordinal encoding")
ordinal = OrdinalEncoder(categories=[["small", "medium", "large"]]).fit(sizes)
print("Ordered size codes:", ordinal.transform(sizes).ravel().astype(int).tolist())

# %% 4. Group rare categories

print("\nStep 4: Group rare categories")
rare = OneHotEncoder(min_frequency=2, handle_unknown="infrequent_if_exist",
                     sparse_output=False).fit(colors)
print(rare.get_feature_names_out().tolist())
print(rare.transform([["green"], ["yellow"], ["purple"]]).astype(int))

# %% 5. Target encoding with cross-fitting

print("\nStep 5: Target encoding with cross-fitting")
examples = pd.DataFrame({"id": ids.ravel(), "target": target})
naive = examples.groupby("id")["target"].transform("mean")
encoder = TargetEncoder(target_type="continuous", smooth=0.0, cv=2,
                        random_state=0)
cross_fitted = encoder.fit_transform(ids, target).ravel()
print("Naive:", naive.tolist())
print("Cross-fitted:", cross_fitted.tolist())
print("Unseen ID:", encoder.transform([["new"]]).ravel().tolist())
