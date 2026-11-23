"""Day 1 | Slides 22 | Three ways information can leak (starter)."""

# %% Setup

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

train = pd.DataFrame({"input": [0, 0, 1, 1, 0, 0, 1, 1],
                      "label": [0, 0, 1, 1, 0, 1, 0, 1]})
test = pd.DataFrame({"input": [0, 0, 0, 1, 1, 1, 1, 0],
                     "label": [0, 0, 0, 1, 1, 0, 1, 1]})

# Fill each TODO in order. Run the file after you add each step.

# An untouched starter only loads the setup and lists the steps.

# %% 1. Target leakage

# TODO Add outcome_copy to both frames and compare a shallow tree's test AUC with
#      and without it.
print("\nStep 1: Target leakage")

# %% 2. Train-test contamination

# TODO Fit a StandardScaler on train_values alone and on the combined arrays.
#      Print both means.
print("\nStep 2: Train-test contamination")

# %% 3. Temporal leakage

# TODO Compare randomly split and chronological arrays of time points 1 through
#      10.
print("\nStep 3: Temporal leakage")
