""" L1.2 - Credit features and preprocessing

We will identify safe inputs, calculate features, split by time and prepare values. This builds the reasoning behind the full credit pipeline.
"""

# %% Setup

import numpy as np
import pandas as pd

# %% 1. Define the chronological split

# All rows in the first group precede the second. Only number and group are available when we predict.

print("\nStep 1 - Define the chronological split")

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

train, test = data[data["time"] <= 8], data[data["time"] > 8]
features = ["number", "group"]
print("Training times:", train["time"].tolist())
print("Testing times:", test["time"].tolist())
print("Selected features:", features)

# %% 2. Log, scale, and encode

# The numeric pipeline compresses large numbers and scales using the training values. The categorical encoder handles a new group in the later rows.

print("\nStep 2 - Log, scale, and encode")

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

# An array of word counts represents text. An image is already a grid of intensity values.

print("\nStep 3 - Text and image as numbers")

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

matrix = CountVectorizer().fit_transform(["red blue", "red green"])
pixels = np.array([[0, 255], [255, 0]])
print("Word-count shape:", matrix.shape)
print("Image pixels:\n", pixels)

# %% 4. Evaluate and revisit the leakage traps

# Print the score and check it against what the model could know. Even a perfect score on four simple rows gives very little evidence. The after-outcome column would just reveal the answer.

print("\nStep 4 - Evaluate and revisit the leakage traps")

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

probabilities = pipeline.predict_proba(test[features])[:, 1]
print("Test AUC on four rows:", round(roc_auc_score(test["label"], probabilities), 2))
print("After-outcome column copies label:", data["after_outcome"].equals(data["label"]))

# %% 5. Domain ratios and disposable income

# Compute LTI with annual income. Compute DBR and remaining income using the same monthly units as the payments.

print("\nStep 5 - Domain ratios and disposable income")

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

payments = values["existing_payment"] + values["new_payment"]
monthly = values["monthly_income"]
ratios = values.assign(lti=values["amount"] / (12 * monthly),
                       dbr=payments / monthly, disposable=monthly - payments)
print(ratios[["lti", "dbr", "disposable"]])

# %% 6. Lag, rolling average, and recency

# Use previous rows for a lag or trailing average. Recency counts how long ago the last available event happened.

print("\nStep 6 - Lag, rolling average, and recency")

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

previous = series.shift(1)
print(pd.DataFrame({"current": series, "lag": previous,
                    "previous_two_mean": previous.rolling(2).mean()}))
cutoff = pd.Timestamp("2025-01-05")
events = pd.to_datetime(["2025-01-01", "2025-01-03", "2025-01-07"])
last_known = events[events <= cutoff].max()
print("Recency in days:", (cutoff - last_known).days)

# %% 7. Interactions and aggregation

# Multiplying two columns creates a new feature. Grouping turns multiple rows into one summary per ID.

print("\nStep 7 - Interactions and aggregation")

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

features = points.assign(total=points["units"] * points["price"])
print(features.groupby("id").agg(count=("total", "size"),
                                sum_total=("total", "sum")))

# %% 8. Filter with correlation and mutual information

# Correlation measures linear association. Mutual information can find a relationship even when a positive and negative slope cancel in correlation.

print("\nStep 8 - Filter with correlation and mutual information")

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

curve = pd.Series(signal ** 2)
print("Correlation:\n", X.corrwith(curve).round(3))
mi = mutual_info_regression(X, curve, random_state=0)
print("Mutual information:\n", pd.Series(mi, index=X.columns).round(3))

# %% 9. Wrapper selection with RFE

# Recursive feature elimination fits a model, removes lower-ranked inputs, and repeats until the requested number is left.

print("\nStep 9 - Wrapper selection with RFE")

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

selector = RFE(LinearRegression(), n_features_to_select=1).fit(X, y)
print("RFE selects:", X.columns[selector.support_].tolist())

# %% 10. Embedded selection with Lasso and a tree

# Lasso can shrink a coefficient to zero during fitting. A tree also learns which features it used to reduce error. We can use those importances to rank features.

print("\nStep 10 - Embedded selection with Lasso and a tree")

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

lasso = Lasso(alpha=0.15).fit(X, y)
tree = DecisionTreeRegressor(max_depth=2, random_state=0).fit(X, y)
print("Lasso coefficients:", dict(zip(X.columns, lasso.coef_.round(3).tolist())))
print("Tree importances:", dict(zip(X.columns, tree.feature_importances_.round(3).tolist())))

# %% 11. Target leakage

# This feature is a copy of the label produced after the outcome. A model gets a perfect score because the answer is in the input.

print("\nStep 11 - Target leakage")

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

train["outcome_copy"] = train["label"]
test["outcome_copy"] = test["label"]
for names in [["input"], ["input", "outcome_copy"]]:
    model = DecisionTreeClassifier(max_depth=1, random_state=0).fit(train[names], train["label"])
    score = roc_auc_score(test["label"], model.predict_proba(test[names])[:, 1])
    print(names, "test AUC:", round(score, 2))

# %% 12. Train-test contamination

# The mean belongs to the training set. Fitting a scaler after combining both sets lets the test set change that mean.

print("\nStep 12 - Train-test contamination")

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

train_values = [[1], [2], [3]]
test_values = [[30], [30]]
honest = StandardScaler().fit(train_values)
contaminated = StandardScaler().fit(train_values + test_values)
print("Training-only mean:", honest.mean_[0])
print("Contaminated mean:", contaminated.mean_[0])

# %% 13. Temporal leakage

# When the model will predict future rows, test on later rows. A shuffled split can train on dates after the ones in the test set.

print("\nStep 13 - Temporal leakage")

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

time = np.arange(1, 11)
random_train, random_test = train_test_split(time, test_size=3, random_state=0)
ordered_train, ordered_test = time[:-3], time[-3:]
print("Shuffled train:", sorted(random_train), "test:", sorted(random_test))
print("Ordered train max:", ordered_train.max(), "test min:", ordered_test.min())
