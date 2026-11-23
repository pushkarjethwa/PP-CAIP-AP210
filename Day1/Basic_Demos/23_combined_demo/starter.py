"""Day 1 | Slides 23 | A small end-to-end run for slide 23 (starter)."""

# %% Setup

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

data = pd.DataFrame({
    "time": range(1, 13),
    "number": [1, 20, 2, 30, 3, 40, 4, 50, 5, 60, 6, 70],
    "group": ["a", "b", "a", "a", "b", "b", "a", "b", "new", "a", "a", "b"],
    "label": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
})
data["after_outcome"] = data["label"]

# Fill each TODO in order. Run the file after you add each step.

# An untouched starter only loads the setup and lists the steps.

# %% 1. Define the chronological split

# TODO Split data at time 8 and list the two features that are available.
print("\nStep 1: Define the chronological split")

# %% 2. Log, scale, and encode

# TODO Build a log-and-scale numeric Pipeline and a one-hot categorical branch in
#      a ColumnTransformer.
print("\nStep 2: Log, scale, and encode")

# %% 3. Text and image as numbers

# TODO Convert two short strings with CountVectorizer and print a 2-by-2 pixel
#      array.
print("\nStep 3: Text and image as numbers")

# %% 4. Evaluate and revisit the leakage traps

# TODO Score the fitted pipeline on test and check whether after_outcome copies
#      the label.
print("\nStep 4: Evaluate and revisit the leakage traps")
