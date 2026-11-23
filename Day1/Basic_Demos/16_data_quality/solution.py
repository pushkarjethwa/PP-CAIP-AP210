"""Day 1 | Slides 16 | The six dimensions of data quality (complete solution)."""

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

# %% 1. Completeness

print("\nStep 1: Completeness")
print("Missing value IDs:", table.loc[table["value"].isna(), "id"].tolist())

# %% 2. Validity

print("\nStep 2: Validity")
present = table["value"].notna()
invalid = present & ~table["value"].between(0, 100)
print("Invalid values:", table.loc[invalid, "value"].tolist())
parsed = pd.to_datetime(["2025-01-10", "01/10/2025"], format="%Y-%m-%d", errors="coerce")
print("Invalid date format:", parsed.isna().tolist())

# %% 3. Consistency

print("\nStep 3: Consistency")
canonical = table["group"].str.strip().str.lower()
print("Before:", table["group"].unique().tolist())
print("After:", canonical.unique().tolist())

# %% 4. Uniqueness

print("\nStep 4: Uniqueness")
print("Repeated IDs:", table.loc[table.duplicated("id", keep=False), "id"].tolist())

# %% 5. Timeliness

print("\nStep 5: Timeliness")
age_days = (snapshot - table["updated"]).dt.days
print("Stale updates:", table.loc[age_days > 7, "updated"].dt.date.tolist())

# %% 6. Accuracy

print("\nStep 6: Accuracy")
disagrees = present & ~invalid & table["value"].ne(table["reference"])
print(table.loc[disagrees, ["id", "value", "reference"]])
