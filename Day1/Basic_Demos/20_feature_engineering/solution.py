"""Day 1 | Slides 20 | Create useful features and select among them (complete solution)."""

# %% Setup

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE, mutual_info_regression
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.tree import DecisionTreeRegressor

values = pd.DataFrame({"amount": [1200, 2400], "monthly_income": [100, 200],
                       "existing_payment": [10, 40], "new_payment": [20, 60]})
series = pd.Series([4, 8, 6, 10], name="value")
points = pd.DataFrame({"id": ["a", "a", "b"], "units": [2, 3, 4],
                       "price": [5, 5, 2]})
rng = np.random.default_rng(42)
signal = np.linspace(-2, 2, 81)
noise = rng.normal(size=len(signal))
X = pd.DataFrame({"signal": signal, "noise": noise})
y = 5 * signal + 0.05 * noise

# %% 1. Domain ratios and disposable income

print("\nStep 1: Domain ratios and disposable income")
payments = values["existing_payment"] + values["new_payment"]
monthly = values["monthly_income"]
ratios = values.assign(lti=values["amount"] / (12 * monthly),
                       dbr=payments / monthly, disposable=monthly - payments)
print(ratios[["lti", "dbr", "disposable"]])

# %% 2. Lag, rolling average, and recency

print("\nStep 2: Lag, rolling average, and recency")
previous = series.shift(1)
print(pd.DataFrame({"current": series, "lag": previous,
                    "previous_two_mean": previous.rolling(2).mean()}))
cutoff = pd.Timestamp("2025-01-05")
events = pd.to_datetime(["2025-01-01", "2025-01-03", "2025-01-07"])
last_known = events[events <= cutoff].max()
print("Recency in days:", (cutoff - last_known).days)

# %% 3. Interactions and aggregation

print("\nStep 3: Interactions and aggregation")
features = points.assign(total=points["units"] * points["price"])
print(features.groupby("id").agg(count=("total", "size"),
                                sum_total=("total", "sum")))

# %% 4. Filter with correlation and mutual information

print("\nStep 4: Filter with correlation and mutual information")
curve = pd.Series(signal ** 2)
print("Correlation:\n", X.corrwith(curve).round(3))
mi = mutual_info_regression(X, curve, random_state=0)
print("Mutual information:\n", pd.Series(mi, index=X.columns).round(3))

# %% 5. Wrapper selection with RFE

print("\nStep 5: Wrapper selection with RFE")
selector = RFE(LinearRegression(), n_features_to_select=1).fit(X, y)
print("RFE selects:", X.columns[selector.support_].tolist())

# %% 6. Embedded selection with Lasso and a tree

print("\nStep 6: Embedded selection with Lasso and a tree")
lasso = Lasso(alpha=0.15).fit(X, y)
tree = DecisionTreeRegressor(max_depth=2, random_state=0).fit(X, y)
print("Lasso coefficients:", dict(zip(X.columns, lasso.coef_.round(3).tolist())))
print("Tree importances:", dict(zip(X.columns, tree.feature_importances_.round(3).tolist())))
