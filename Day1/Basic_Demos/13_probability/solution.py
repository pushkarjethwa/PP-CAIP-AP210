"""Day 1 | Slides 13 | Base rate, conditional probability, lift, Bayes, and causation (complete solution)."""

# %% Setup

import pandas as pd

table = pd.DataFrame({
    "X": [True, True, True, True, False, False, False, False],
    "Y": [True, True, True, False, True, False, False, False],
})

# %% 1. Base rate

print("\nStep 1: Base rate")
base = table["Y"].mean()
print("P(Y):", base)

# %% 2. Conditional probability and lift

print("\nStep 2: Conditional probability and lift")
y_given_x = table.loc[table["X"], "Y"].mean()
print("P(Y given X):", y_given_x, "Lift:", y_given_x / base)

# %% 3. Bayes' rule

print("\nStep 3: Bayes' rule")
bayes = y_given_x * table["X"].mean() / base
direct = table.loc[table["Y"], "X"].mean()
print("P(X given Y), Bayes:", bayes, "Direct count:", direct)

# %% 4. Correlation and cause

print("\nStep 4: Correlation and cause")
z = pd.Series([1, 2, 3, 4])
a, b = 2 * z, 3 * z
print("Correlation:", round(a.corr(b), 2))
a = a + 10
print("b after changing a:", b.tolist())
