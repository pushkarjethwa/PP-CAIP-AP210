"""Day 1 | Slides 15 | A seven-step preview of Lab L1.1 (starter)."""

# %% Setup

from io import BytesIO, StringIO

import pandas as pd

csv_text = """id,group,score,updated,email,future_flag
1, a ,9.50,2025-01-01,a@example.invalid,1
1,A,10.50,2025/01/03,a@example.invalid,1
2,b,200,02 Jan 2025,b@example.invalid,0
3,B,,2025-01-02,c@example.invalid,0
4,c,12,2025/01/02,d@example.invalid,1
"""

# Fill each TODO in order. Run the file after you add each step.

# An untouched starter only loads the setup and lists the steps.

# %% 1. Load and profile

# TODO Read csv_text and record the row, missing-score, and duplicate-ID counts.
print("\nStep 1: Load and profile")

# %% 2. Remove sensitive and future columns

# TODO Drop email and future_flag from raw and print the remaining columns.
print("\nStep 2: Remove sensitive and future columns")

# %% 3. Standardise categories

# TODO Apply str.strip() and str.lower() to group.
print("\nStep 3: Standardise categories")

# %% 4. Parse numbers and dates

# TODO Convert scores with to_numeric and parse updated with
#      to_datetime(format='mixed').
print("\nStep 4: Parse numbers and dates")

# %% 5. Validate range and remove repeated IDs

# TODO Keep scores between 0 and 100 with where(), then sort by date and
#      deduplicate id.
print("\nStep 5: Validate range and remove repeated IDs")

# %% 6. Flag and fill missing scores

# TODO Add a score_missing indicator and fill missing scores with the median.
print("\nStep 6: Flag and fill missing scores")

# %% 7. Validate and create the report

# TODO Convert group to category, assert the rules, build a comparison report, and
#      write Parquet to BytesIO.
print("\nStep 7: Validate and create the report")
