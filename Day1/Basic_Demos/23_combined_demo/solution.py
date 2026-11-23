"""Day 1 | Slides 23 | A small end-to-end run for slide 23 (complete solution)."""

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

# %% 1. Define the chronological split

print("\nStep 1: Define the chronological split")
train, test = data[data["time"] <= 8], data[data["time"] > 8]
features = ["number", "group"]
print("Training times:", train["time"].tolist())
print("Testing times:", test["time"].tolist())
print("Selected features:", features)

# %% 2. Log, scale, and encode

print("\nStep 2: Log, scale, and encode")
numbers = Pipeline([("log", FunctionTransformer(np.log1p)), ("scale", StandardScaler())])
preprocess = ColumnTransformer([
    ("number", numbers, ["number"]),
    ("group", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["group"]),
])
pipeline = Pipeline([("preprocess", preprocess), ("model", LogisticRegression())])
pipeline.fit(train[features], train["label"])
transformed = pipeline.named_steps["preprocess"].transform(test[features])
print("Later matrix shape:", transformed.shape)
print("Unseen group's indicators:", transformed[0, 1:].tolist())

# %% 3. Text and image as numbers

print("\nStep 3: Text and image as numbers")
matrix = CountVectorizer().fit_transform(["red blue", "red green"])
pixels = np.array([[0, 255], [255, 0]])
print("Word-count shape:", matrix.shape)
print("Image pixels:\n", pixels)

# %% 4. Evaluate and revisit the leakage traps

print("\nStep 4: Evaluate and revisit the leakage traps")
probabilities = pipeline.predict_proba(test[features])[:, 1]
print("Test AUC on four rows:", round(roc_auc_score(test["label"], probabilities), 2))
print("After-outcome column copies label:", data["after_outcome"].equals(data["label"]))
