"""Day 1 | Slides 10 | Exploratory data analysis, one check at a time (complete solution)."""

# %% Setup

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

# %% 1. Load, size, and preview

print("\nStep 1: Load, size, and preview")
print("Rows, columns:", df.shape)
print("Column names:", df.columns.tolist())
print(df.head(3))

# %% 2. Types and memory

print("\nStep 2: Types and memory")
df.info()
print("Total memory in bytes:", df.memory_usage(deep=True).sum())

# %% 3. Reduce memory for repeated text

print("\nStep 3: Reduce memory for repeated text")
repeated = pd.concat([df["group"]] * 100, ignore_index=True)
category = repeated.astype("category")
print("Text bytes:", repeated.memory_usage(index=False, deep=True))
print("Category bytes:", category.memory_usage(index=False, deep=True))

# %% 4. Missing values

print("\nStep 4: Missing values")
print("Missing counts:\n", df.isna().sum())
print("Missing percent:\n", (100 * df.isna().mean()).round(1))

# %% 5. Duplicates

print("\nStep 5: Duplicates")
print("Duplicate rows:", df.duplicated().sum())
print("Duplicate IDs after the first:", df["id"].duplicated().sum())

# %% 6. Categorical counts

print("\nStep 6: Categorical counts")
print(df["group"].value_counts(dropna=False))

# %% 7. Numeric summary

print("\nStep 7: Numeric summary")
print(df[["x", "y"]].describe().round(2))
print("x skew:", round(df["x"].skew(), 2))

# %% 8. Validity and possible outliers

print("\nStep 8: Validity and possible outliers")
print("Negative x:\n", df.loc[df["x"] < 0, ["id", "x"]])
q1, q3 = df["x"].quantile([0.25, 0.75])
upper = q3 + 1.5 * (q3 - q1)
print("Above upper IQR fence:\n", df.loc[df["x"] > upper, ["id", "x"]])

# %% 9. Group summaries and correlation

print("\nStep 9: Group summaries and correlation")
print("Group means:\n", df.groupby("group")[["x", "y"]].mean().round(2))
print("Correlation:\n", df[["x", "y"]].corr().round(2))

# %% 10. Histogram, box plot, and scatter plot

print("\nStep 10: Histogram, box plot, and scatter plot")
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
