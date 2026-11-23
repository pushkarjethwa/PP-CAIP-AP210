"""Day 1 | Slides 15 | A seven-step preview of Lab L1.1 (complete solution)."""

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

# %% 1. Load and profile

print("\nStep 1: Load and profile")
raw = pd.read_csv(StringIO(csv_text), dtype={"score": "string"})
before = {"rows": len(raw), "missing_score": int(raw["score"].isna().sum()),
          "duplicate_ids": int(raw["id"].duplicated().sum())}
print("Before:", before)

# %% 2. Remove sensitive and future columns

print("\nStep 2: Remove sensitive and future columns")
clean = raw.drop(columns=["email", "future_flag"])
print("Working columns:", clean.columns.tolist())

# %% 3. Standardise categories

print("\nStep 3: Standardise categories")
clean["group"] = clean["group"].str.strip().str.lower()
print("Groups:", sorted(clean["group"].unique()))

# %% 4. Parse numbers and dates

print("\nStep 4: Parse numbers and dates")
clean["score"] = pd.to_numeric(clean["score"], errors="coerce")
clean["updated"] = pd.to_datetime(clean["updated"], format="mixed", errors="coerce")
print(clean.dtypes)

# %% 5. Validate range and remove repeated IDs

print("\nStep 5: Validate range and remove repeated IDs")
clean["score"] = clean["score"].where(clean["score"].between(0, 100))
clean = clean.sort_values("updated").drop_duplicates("id", keep="last").copy()
print(clean[["id", "score", "updated"]].sort_values("id"))

# %% 6. Flag and fill missing scores

print("\nStep 6: Flag and fill missing scores")
clean["score_missing"] = clean["score"].isna()
clean["score"] = clean["score"].fillna(clean["score"].median())
print(clean[["id", "score", "score_missing"]].sort_values("id"))

# %% 7. Validate and create the report

print("\nStep 7: Validate and create the report")
clean["group"] = clean["group"].astype("category")
assert clean["id"].is_unique and clean["score"].between(0, 100).all()
assert clean["updated"].notna().all()
after = {"rows": len(clean), "missing_score": int(clean["score"].isna().sum()),
         "duplicate_ids": int(clean["id"].duplicated().sum())}
report = pd.DataFrame({"before": before, "after": after}).rename_axis("check")
print(report.to_csv(), end="")
print("Scores filled:", clean["score_missing"].sum())
buffer = BytesIO()
clean.to_parquet(buffer, engine="pyarrow", index=False)
print("Clean rows read back:", len(pd.read_parquet(buffer, engine="pyarrow")))
