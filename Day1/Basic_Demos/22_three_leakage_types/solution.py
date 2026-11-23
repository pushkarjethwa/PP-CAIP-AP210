"""Day 1 | Slides 22 | Three ways information can leak (complete solution)."""

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

# %% 1. Target leakage

print("\nStep 1: Target leakage")
train["outcome_copy"] = train["label"]
test["outcome_copy"] = test["label"]
for names in [["input"], ["input", "outcome_copy"]]:
    model = DecisionTreeClassifier(max_depth=1, random_state=0).fit(train[names], train["label"])
    score = roc_auc_score(test["label"], model.predict_proba(test[names])[:, 1])
    print(names, "test AUC:", round(score, 2))

# %% 2. Train-test contamination

print("\nStep 2: Train-test contamination")
train_values = [[1], [2], [3]]
test_values = [[30], [30]]
honest = StandardScaler().fit(train_values)
contaminated = StandardScaler().fit(train_values + test_values)
print("Training-only mean:", honest.mean_[0])
print("Contaminated mean:", contaminated.mean_[0])

# %% 3. Temporal leakage

print("\nStep 3: Temporal leakage")
time = np.arange(1, 11)
random_train, random_test = train_test_split(time, test_size=3, random_state=0)
ordered_train, ordered_test = time[:-3], time[-3:]
print("Shuffled train:", sorted(random_train), "test:", sorted(random_test))
print("Ordered train max:", ordered_train.max(), "test min:", ordered_test.min())
