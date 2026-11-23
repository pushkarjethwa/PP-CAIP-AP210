"""Day 1 | Slides 13 | Base rate, conditional probability, lift, Bayes, and causation (starter)."""

# %% Setup

import pandas as pd

table = pd.DataFrame({
    "X": [True, True, True, True, False, False, False, False],
    "Y": [True, True, True, False, True, False, False, False],
})

# Fill each TODO in order. Run the file after you add each step.

# An untouched starter only loads the setup and lists the steps.

# %% 1. Base rate

# TODO Find the mean of the Y column and save it as base.
print("\nStep 1: Base rate")

# %% 2. Conditional probability and lift

# TODO Filter table to X, take the mean of Y, and divide by base.
print("\nStep 2: Conditional probability and lift")

# %% 3. Bayes' rule

# TODO Calculate y_given_x * P(X) / base and compare against the mean of X among Y
#      rows.
print("\nStep 3: Bayes' rule")

# %% 4. Correlation and cause

# TODO Make a = 2 * z and b = 3 * z, compute their correlation, then replace a.
print("\nStep 4: Correlation and cause")
