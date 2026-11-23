"""Day 1 | Slides 10 | Exploratory data analysis, one check at a time (starter)."""

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

# Fill each TODO in order. Run the file after you add each step.

# An untouched starter only loads the setup and lists the steps.

# %% 1. Load, size, and preview

# TODO Print df.shape, df.columns.tolist(), and df.head(3).
print("\nStep 1: Load, size, and preview")

# %% 2. Types and memory

# TODO Call df.info() and sum the result of df.memory_usage(deep=True).
print("\nStep 2: Types and memory")

# %% 3. Reduce memory for repeated text

# TODO Repeat the group column and compare memory_usage before and after
#      astype('category').
print("\nStep 3: Reduce memory for repeated text")

# %% 4. Missing values

# TODO Print isna().sum() and isna().mean() multiplied by 100.
print("\nStep 4: Missing values")

# %% 5. Duplicates

# TODO Count duplicates for whole rows and for the id column.
print("\nStep 5: Duplicates")

# %% 6. Categorical counts

# TODO Print value_counts on the group column and include missing values.
print("\nStep 6: Categorical counts")

# %% 7. Numeric summary

# TODO Print describe for x and y, then print the skew of x.
print("\nStep 7: Numeric summary")

# %% 8. Validity and possible outliers

# TODO Print negative x rows, compute Q1 and Q3, and show rows above Q3 + 1.5 ×
#      IQR.
print("\nStep 8: Validity and possible outliers")

# %% 9. Group summaries and correlation

# TODO Calculate mean x and y by group and print their correlation matrix.
print("\nStep 9: Group summaries and correlation")

# %% 10. Histogram, box plot, and scatter plot

# TODO If --plot is present, build axes for a histogram, box plot, and scatter
#      plot.
print("\nStep 10: Histogram, box plot, and scatter plot")
