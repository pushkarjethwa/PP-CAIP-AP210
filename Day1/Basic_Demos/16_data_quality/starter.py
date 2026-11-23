"""Day 1 | Slides 16 | The six dimensions of data quality (starter)."""

# %% Setup

import pandas as pd

table = pd.DataFrame({
    "id": [1, 2, 2, 3, 4],
    "value": [12, 200, 20, None, 40],
    "reference": [10, 200, 20, 30, 40],
    "group": ["red", " BLUE ", "blue", "red", "green"],
    "updated": pd.to_datetime(["2025-01-09", "2024-12-20", "2025-01-08",
                               "2025-01-07", "2025-01-06"]),
})
snapshot = pd.Timestamp("2025-01-10")

# Fill each TODO in order. Run the file after you add each step.

# An untouched starter only loads the setup and lists the steps.

# %% 1. Completeness

# TODO Find IDs where value.isna() is True.
print("\nStep 1: Completeness")

# %% 2. Validity

# TODO Find present values outside 0 to 100, then check two strings against a
#      required date format.
print("\nStep 2: Validity")

# %% 3. Consistency

# TODO Compare the unique groups before and after stripping spaces and
#      lowercasing.
print("\nStep 3: Consistency")

# %% 4. Uniqueness

# TODO Use duplicated('id', keep=False) and print the flagged IDs.
print("\nStep 4: Uniqueness")

# %% 5. Timeliness

# TODO Subtract updated from snapshot, and print dates more than seven days old.
print("\nStep 5: Timeliness")

# %% 6. Accuracy

# TODO Filter present, valid values that differ from reference and print those
#      rows.
print("\nStep 6: Accuracy")
