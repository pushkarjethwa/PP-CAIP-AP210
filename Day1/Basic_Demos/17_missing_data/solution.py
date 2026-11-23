"""Day 1 | Slides 17 | Why values go missing and how to handle them (complete solution)."""

# %% Setup

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(42)
full = pd.DataFrame({"group": ["a", "a", "b", "b", "a", "b"],
                     "value": [2, 4, 6, 8, 10, 12]})
incomplete = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6],
                           "group": ["a", "a", "a", "b", "b", "b"],
                           "value": [2, np.nan, 6, 8, np.nan, 12]})

# %% 1. MCAR

print("\nStep 1: MCAR")
mcar = full["value"].copy().astype(float)
random_rows = rng.choice(len(full), size=2, replace=False)
mcar.iloc[random_rows] = np.nan
print("MCAR:", mcar.tolist())

# %% 2. MAR

print("\nStep 2: MAR")
print("MAR:", full["value"].mask(full["group"].eq("b")).tolist())

# %% 3. MNAR

print("\nStep 3: MNAR")
print("MNAR:", full["value"].mask(full["value"] > 8).tolist())

# %% 4. Drop rows or fill a simple median

print("\nStep 4: Drop rows or fill a simple median")
print("Rows after dropping:", len(incomplete.dropna(subset=["value"])))
values = incomplete["value"]
print("Median fill:", values.fillna(values.median()).tolist())

# %% 5. Fill using groups and keep an indicator

print("\nStep 5: Fill using groups and keep an indicator")
missing = values.isna()
group_medians = incomplete.groupby("group")["value"].transform("median")
print(incomplete.assign(value_filled=values.fillna(group_medians),
                        value_missing=missing))

# %% 6. Model-based fill

print("\nStep 6: Model-based fill")
imputer = LinearRegression().fit(incomplete.loc[~missing, ["x"]], values[~missing])
predicted = imputer.predict(incomplete.loc[missing, ["x"]])
print("Model fills:", predicted.round(2).tolist())
