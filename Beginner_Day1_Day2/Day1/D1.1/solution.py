""" D1.1 - Problem framing and data exploration

We will turn a question into a prediction task and inspect a small table. This shows what the data contains before anyone builds a model.
"""

# %% Setup

import numpy as np
import pandas as pd

# %% 1. Build one row per time point

# One row is a time point. The current and previous readings are available by that point.

print("\nStep 1 - Build one row per time point")

import pandas as pd
from sklearn.metrics import mean_absolute_error

series = pd.Series([10, 13, 12, 16, 15], name="reading")

rows = series.to_frame().assign(previous=series.shift(1))
print(rows)

# %% 2. Define a one-step target

# The prediction horizon is one step. We can build the label from the next reading while preparing historical examples.

print("\nStep 2 - Define a one-step target")

import pandas as pd
from sklearn.metrics import mean_absolute_error

series = pd.Series([10, 13, 12, 16, 15], name="reading")

rows["next_reading"] = series.shift(-1)
print(rows)
labelled = rows.dropna(subset=["previous", "next_reading"])

# %% 3. Score a baseline

# Suppose we predict that the next reading equals the current one. Mean absolute error measures the typical size of the miss in the original units.

print("\nStep 3 - Score a baseline")

import pandas as pd
from sklearn.metrics import mean_absolute_error

series = pd.Series([10, 13, 12, 16, 15], name="reading")

error = mean_absolute_error(labelled["next_reading"], labelled["reading"])
print("Baseline MAE:", error)

# %% 4. Array, shape, dtype, and boolean mask

# Shape tells us the size of each axis. A comparison gives one True or False per value. The mask keeps only True entries.

print("\nStep 4 - Array, shape, dtype, and boolean mask")

from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

values = np.array([10, 20, 30])
big = np.arange(300_000)
table = pd.DataFrame({"id": [1, 2, 3, 4], "group": ["a", "a", "b", "b"],
                      "value": [2, 4, 6, 8]})
labels = pd.DataFrame({"group": ["a", "b"], "name": ["alpha", "beta"]})
train_x = [[1], [2], [3]]
train_y = [2, 4, 6]

print("Shape:", values.shape, "dtype:", values.dtype)
mask = values >= 20
print("Mask:", mask.tolist(), "Selected:", values[mask].tolist())

# %% 5. Vectorized maths vs a Python loop

# The operation is doubling. The loop calls Python for each value. The array expression passes the job to NumPy.

print("\nStep 5 - Vectorized maths vs a Python loop")

from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

values = np.array([10, 20, 30])
big = np.arange(300_000)
table = pd.DataFrame({"id": [1, 2, 3, 4], "group": ["a", "a", "b", "b"],
                      "value": [2, 4, 6, 8]})
labels = pd.DataFrame({"group": ["a", "b"], "name": ["alpha", "beta"]})
train_x = [[1], [2], [3]]
train_y = [2, 4, 6]

start = perf_counter()
doubled_loop = []
for value in big:
    doubled_loop.append(value * 2)
loop_seconds = perf_counter() - start
start = perf_counter()
doubled_array = big * 2
array_seconds = perf_counter() - start
print("Same values:", np.array_equal(doubled_loop, doubled_array))
print(f"Loop: {loop_seconds:.5f}s | Array: {array_seconds:.5f}s")

# %% 6. Load, size, and preview

# Start by confirming what came in. A CSV read creates a DataFrame.

print("\nStep 6 - Load, size, and preview")

from io import StringIO
import sys

import pandas as pd

csv_text = """id,group,x,y
1,a,10,2
2,A,11,3
3,b,12,4
3,b,12,4
4,b,,5
5,a,14,6
6,b,100,10
7,c,16,7
8,c,-2,8
"""
df = pd.read_csv(StringIO(csv_text))

print("Rows, columns:", df.shape)
print("Column names:", df.columns.tolist())
print(df.head(3))

# %% 7. Types and memory

# Check which columns are numeric and how much memory the DataFrame uses.

print("\nStep 7 - Types and memory")

from io import StringIO
import sys

import pandas as pd

csv_text = """id,group,x,y
1,a,10,2
2,A,11,3
3,b,12,4
3,b,12,4
4,b,,5
5,a,14,6
6,b,100,10
7,c,16,7
8,c,-2,8
"""
df = pd.read_csv(StringIO(csv_text))

df.info()
print("Total memory in bytes:", df.memory_usage(deep=True).sum())

# %% 8. Reduce memory for repeated text

# When the same category labels repeat, pandas can store a short code per row and keep the text once.

print("\nStep 8 - Reduce memory for repeated text")

from io import StringIO
import sys

import pandas as pd

csv_text = """id,group,x,y
1,a,10,2
2,A,11,3
3,b,12,4
3,b,12,4
4,b,,5
5,a,14,6
6,b,100,10
7,c,16,7
8,c,-2,8
"""
df = pd.read_csv(StringIO(csv_text))

repeated = pd.concat([df["group"]] * 100, ignore_index=True)
category = repeated.astype("category")
print("Text bytes:", repeated.memory_usage(index=False, deep=True))
print("Category bytes:", category.memory_usage(index=False, deep=True))

# %% 9. Missing values

# Count blanks and divide by the row count to compare columns.

print("\nStep 9 - Missing values")

from io import StringIO
import sys

import pandas as pd

csv_text = """id,group,x,y
1,a,10,2
2,A,11,3
3,b,12,4
3,b,12,4
4,b,,5
5,a,14,6
6,b,100,10
7,c,16,7
8,c,-2,8
"""
df = pd.read_csv(StringIO(csv_text))

print("Missing counts:\n", df.isna().sum())
print("Missing percent:\n", (100 * df.isna().mean()).round(1))

# %% 10. Duplicates

# Check for repeated rows and repeated IDs. Those answer different questions.

print("\nStep 10 - Duplicates")

from io import StringIO
import sys

import pandas as pd

csv_text = """id,group,x,y
1,a,10,2
2,A,11,3
3,b,12,4
3,b,12,4
4,b,,5
5,a,14,6
6,b,100,10
7,c,16,7
8,c,-2,8
"""
df = pd.read_csv(StringIO(csv_text))

print("Duplicate rows:", df.duplicated().sum())
print("Duplicate IDs after the first:", df["id"].duplicated().sum())

# %% 11. Categorical counts

# Frequency counts can surface a spelling or case issue.

print("\nStep 11 - Categorical counts")

from io import StringIO
import sys

import pandas as pd

csv_text = """id,group,x,y
1,a,10,2
2,A,11,3
3,b,12,4
3,b,12,4
4,b,,5
5,a,14,6
6,b,100,10
7,c,16,7
8,c,-2,8
"""
df = pd.read_csv(StringIO(csv_text))

print(df["group"].value_counts(dropna=False))

# %% 12. Numeric summary

# Use the summary and quantiles to get a quick sense of the centre, spread, and extremes.

print("\nStep 12 - Numeric summary")

from io import StringIO
import sys

import pandas as pd

csv_text = """id,group,x,y
1,a,10,2
2,A,11,3
3,b,12,4
3,b,12,4
4,b,,5
5,a,14,6
6,b,100,10
7,c,16,7
8,c,-2,8
"""
df = pd.read_csv(StringIO(csv_text))

print(df[["x", "y"]].describe().round(2))
print("x skew:", round(df["x"].skew(), 2))

# %% 13. Validity and possible outliers

# A known range can flag invalid values. An IQR rule flags values to investigate.

print("\nStep 13 - Validity and possible outliers")

from io import StringIO
import sys

import pandas as pd

csv_text = """id,group,x,y
1,a,10,2
2,A,11,3
3,b,12,4
3,b,12,4
4,b,,5
5,a,14,6
6,b,100,10
7,c,16,7
8,c,-2,8
"""
df = pd.read_csv(StringIO(csv_text))

print("Negative x:\n", df.loc[df["x"] < 0, ["id", "x"]])
q1, q3 = df["x"].quantile([0.25, 0.75])
upper = q3 + 1.5 * (q3 - q1)
print("Above upper IQR fence:\n", df.loc[df["x"] > upper, ["id", "x"]])

# %% 14. Group summaries and correlation

# Compare groups with a simple summary. Then use correlation to check for a linear association between two numeric columns.

print("\nStep 14 - Group summaries and correlation")

from io import StringIO
import sys

import pandas as pd

csv_text = """id,group,x,y
1,a,10,2
2,A,11,3
3,b,12,4
3,b,12,4
4,b,,5
5,a,14,6
6,b,100,10
7,c,16,7
8,c,-2,8
"""
df = pd.read_csv(StringIO(csv_text))

print("Group means:\n", df.groupby("group")[["x", "y"]].mean().round(2))
print("Correlation:\n", df[["x", "y"]].corr().round(2))

# %% 15. Histogram, box plot, and scatter plot

# A histogram shows the shape, a box plot flags extremes, and a scatter plot shows pairs of values.

print("\nStep 15 - Histogram, box plot, and scatter plot")

from io import StringIO
import sys

import pandas as pd

csv_text = """id,group,x,y
1,a,10,2
2,A,11,3
3,b,12,4
3,b,12,4
4,b,,5
5,a,14,6
6,b,100,10
7,c,16,7
8,c,-2,8
"""
df = pd.read_csv(StringIO(csv_text))

if "--plot" in sys.argv:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    axes[0].hist(df["x"].dropna(), bins=8)
    axes[0].set(xlabel="x", ylabel="Count", title="Histogram")
    axes[1].boxplot(df["x"].dropna())
    axes[1].set(ylabel="x", title="Box plot")
    axes[2].scatter(df["x"], df["y"])
    axes[2].set(xlabel="x", ylabel="y", title="Scatter plot")
    fig.tight_layout()
    plt.show()

# %% 16. Mean and median

# The mean uses every magnitude. The median sits at the middle of the sorted values.

print("\nStep 16 - Mean and median")

import sys

import numpy as np
import pandas as pd

values = pd.Series([1, 2, 2, 3, 12])
rng = np.random.default_rng(42)

print("Mean:", values.mean(), "Median:", values.median())

# %% 17. Variance and standard deviation

# Both describe spread. Standard deviation is the square root of variance and is in the same units as the values.

print("\nStep 17 - Variance and standard deviation")

import sys

import numpy as np
import pandas as pd

values = pd.Series([1, 2, 2, 3, 12])
rng = np.random.default_rng(42)

print("Sample variance:", round(values.var(), 2))
print("Sample standard deviation:", round(values.std(), 2))

# %% 18. Skew and a log

# The large value makes a right tail. A log compresses the higher values more than the lower ones.

print("\nStep 18 - Skew and a log")

import sys

import numpy as np
import pandas as pd

values = pd.Series([1, 2, 2, 3, 12])
rng = np.random.default_rng(42)

print("Skew before log:", round(values.skew(), 2))
print("Skew after log:", round(np.log(values).skew(), 2))

# %% 19. Base rate

# The base rate uses all eight rows. The mean of a Boolean column is its True fraction.

print("\nStep 19 - Base rate")

import pandas as pd

table = pd.DataFrame({
    "X": [True, True, True, True, False, False, False, False],
    "Y": [True, True, True, False, True, False, False, False],
})

base = table["Y"].mean()
print("P(Y):", base)

# %% 20. Conditional probability and lift

# Keep the rows where X is true. The fraction of Y in these rows answers P(Y given X). Lift divides that fraction by the base rate.

print("\nStep 20 - Conditional probability and lift")

import pandas as pd

table = pd.DataFrame({
    "X": [True, True, True, True, False, False, False, False],
    "Y": [True, True, True, False, True, False, False, False],
})

y_given_x = table.loc[table["X"], "Y"].mean()
print("P(Y given X):", y_given_x, "Lift:", y_given_x / base)

# %% 21. Bayes' rule

# Reverse the condition using P(Y given X), P(X), and P(Y). Then check it by counting directly.

print("\nStep 21 - Bayes' rule")

import pandas as pd

table = pd.DataFrame({
    "X": [True, True, True, True, False, False, False, False],
    "Y": [True, True, True, False, True, False, False, False],
})

bayes = y_given_x * table["X"].mean() / base
direct = table.loc[table["Y"], "X"].mean()
print("P(X given Y), Bayes:", bayes, "Direct count:", direct)

# %% 22. Vector and dot product

# Multiply matching entries, add the products, and then add the intercept.

print("\nStep 22 - Vector and dot product")

import sys

import numpy as np
from sklearn.metrics import log_loss

x = np.array([2, 3])
w = np.array([4, 1])
b = 1

score = x @ w
print("Dot product:", score, "Prediction:", score + b)

# %% 23. Mean squared error

# Subtract actual values from predictions, square each error, and average.

print("\nStep 23 - Mean squared error")

import sys

import numpy as np
from sklearn.metrics import log_loss

x = np.array([2, 3])
w = np.array([4, 1])
b = 1

squared_errors = (np.array([2, 7]) - np.array([1, 5])) ** 2
print("Squared errors:", squared_errors.tolist(), "MSE:", squared_errors.mean())

# %% 24. Gradient descent and learning rate

# At each step, subtract the gradient times the learning rate. Watch where the weight goes when I make the step too large.

print("\nStep 24 - Gradient descent and learning rate")

import sys

import numpy as np
from sklearn.metrics import log_loss

x = np.array([2, 3])
w = np.array([4, 1])
b = 1

paths = {}
for rate in [0.25, 1.1]:
    weight = 0.0
    weights = [weight]
    for _ in range(4):
        gradient = 2 * (weight - 3)
        weight -= rate * gradient
        weights.append(weight)
    paths[rate] = np.array(weights)
    print("Rate:", rate, "Weights:", paths[rate].round(2),
          "Losses:", ((paths[rate] - 3) ** 2).round(2))
