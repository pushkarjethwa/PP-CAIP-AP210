"""Day 1 | Slides 20 | Create useful features and select among them (starter)."""

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

# Fill each TODO in order. Run the file after you add each step.

# An untouched starter only loads the setup and lists the steps.

# %% 1. Domain ratios and disposable income

# TODO Create loan-to-income, payment-to-income (DBR), and disposable income
#      columns.
print("\nStep 1: Domain ratios and disposable income")

# %% 2. Lag, rolling average, and recency

# TODO Build a previous-value lag, a previous-two mean, and recency from events
#      dated no later than the cutoff.
print("\nStep 2: Lag, rolling average, and recency")

# %% 3. Interactions and aggregation

# TODO Multiply units by price, then group by ID to count rows and sum the
#      product.
print("\nStep 3: Interactions and aggregation")

# %% 4. Filter with correlation and mutual information

# TODO Compare each feature with signal squared using corrwith and
#      mutual_info_regression.
print("\nStep 4: Filter with correlation and mutual information")

# %% 5. Wrapper selection with RFE

# TODO Use RFE with LinearRegression and n_features_to_select=1.
print("\nStep 5: Wrapper selection with RFE")

# %% 6. Embedded selection with Lasso and a tree

# TODO Fit Lasso and a shallow tree and print coefficients and feature
#      importances.
print("\nStep 6: Embedded selection with Lasso and a tree")
