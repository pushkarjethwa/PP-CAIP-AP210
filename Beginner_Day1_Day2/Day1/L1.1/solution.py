""" L1.1 - Customer data quality

We will clean a small table in manageable steps. Each step exposes a different data-quality decision before learners see the full lab.
"""

# %% Setup

import numpy as np
import pandas as pd

# %% 1. Load and profile

# Check row count, missing values, and duplicates before touching the data.

print("\nStep 1 - Load and profile")

from io import BytesIO, StringIO

import pandas as pd

csv_text = """id,group,score,updated,email,future_flag
1, a ,9.50,2025-01-01,a@example.invalid,1
1,A,10.50,2025/01/03,a@example.invalid,1
2,b,200,02 Jan 2025,b@example.invalid,0
3,B,,2025-01-02,c@example.invalid,0
4,c,12,2025/01/02,d@example.invalid,1
"""

raw = pd.read_csv(StringIO(csv_text), dtype={"score": "string"})
before = {"rows": len(raw), "missing_score": int(raw["score"].isna().sum()),
          "duplicate_ids": int(raw["id"].duplicated().sum())}
print("Before:", before)

# %% 2. Remove sensitive and future columns

# Email identifies a person. The future flag is unavailable at the time the features are needed. Drop both for their respective reasons.

print("\nStep 2 - Remove sensitive and future columns")

from io import BytesIO, StringIO

import pandas as pd

csv_text = """id,group,score,updated,email,future_flag
1, a ,9.50,2025-01-01,a@example.invalid,1
1,A,10.50,2025/01/03,a@example.invalid,1
2,b,200,02 Jan 2025,b@example.invalid,0
3,B,,2025-01-02,c@example.invalid,0
4,c,12,2025/01/02,d@example.invalid,1
"""

clean = raw.drop(columns=["email", "future_flag"])
print("Working columns:", clean.columns.tolist())

# %% 3. Standardise categories

# Trim spaces and use one case so the same group has the same value.

print("\nStep 3 - Standardise categories")

from io import BytesIO, StringIO

import pandas as pd

csv_text = """id,group,score,updated,email,future_flag
1, a ,9.50,2025-01-01,a@example.invalid,1
1,A,10.50,2025/01/03,a@example.invalid,1
2,b,200,02 Jan 2025,b@example.invalid,0
3,B,,2025-01-02,c@example.invalid,0
4,c,12,2025/01/02,d@example.invalid,1
"""

clean["group"] = clean["group"].str.strip().str.lower()
print("Groups:", sorted(clean["group"].unique()))

# %% 4. Parse numbers and dates

# A text representation needs to become a number or a date before we can compare it consistently.

print("\nStep 4 - Parse numbers and dates")

from io import BytesIO, StringIO

import pandas as pd

csv_text = """id,group,score,updated,email,future_flag
1, a ,9.50,2025-01-01,a@example.invalid,1
1,A,10.50,2025/01/03,a@example.invalid,1
2,b,200,02 Jan 2025,b@example.invalid,0
3,B,,2025-01-02,c@example.invalid,0
4,c,12,2025/01/02,d@example.invalid,1
"""

clean["score"] = pd.to_numeric(clean["score"], errors="coerce")
clean["updated"] = pd.to_datetime(clean["updated"], format="mixed", errors="coerce")
print(clean.dtypes)

# %% 5. Validate range and remove repeated IDs

# Mark out-of-range scores as missing. Keep the most recently updated row for each ID.

print("\nStep 5 - Validate range and remove repeated IDs")

from io import BytesIO, StringIO

import pandas as pd

csv_text = """id,group,score,updated,email,future_flag
1, a ,9.50,2025-01-01,a@example.invalid,1
1,A,10.50,2025/01/03,a@example.invalid,1
2,b,200,02 Jan 2025,b@example.invalid,0
3,B,,2025-01-02,c@example.invalid,0
4,c,12,2025/01/02,d@example.invalid,1
"""

clean["score"] = clean["score"].where(clean["score"].between(0, 100))
clean = clean.sort_values("updated").drop_duplicates("id", keep="last").copy()
print(clean[["id", "score", "updated"]].sort_values("id"))

# %% 6. Flag and fill missing scores

# Save a missingness flag first so we can still see which values needed a fill.

print("\nStep 6 - Flag and fill missing scores")

from io import BytesIO, StringIO

import pandas as pd

csv_text = """id,group,score,updated,email,future_flag
1, a ,9.50,2025-01-01,a@example.invalid,1
1,A,10.50,2025/01/03,a@example.invalid,1
2,b,200,02 Jan 2025,b@example.invalid,0
3,B,,2025-01-02,c@example.invalid,0
4,c,12,2025/01/02,d@example.invalid,1
"""

clean["score_missing"] = clean["score"].isna()
clean["score"] = clean["score"].fillna(clean["score"].median())
print(clean[["id", "score", "score_missing"]].sort_values("id"))

# %% 7. Validate and create the report

# Check that IDs are unique, scores are in range, and dates parsed. Save the before-and-after counts so someone can reproduce the check.

print("\nStep 7 - Validate and create the report")

from io import BytesIO, StringIO

import pandas as pd

csv_text = """id,group,score,updated,email,future_flag
1, a ,9.50,2025-01-01,a@example.invalid,1
1,A,10.50,2025/01/03,a@example.invalid,1
2,b,200,02 Jan 2025,b@example.invalid,0
3,B,,2025-01-02,c@example.invalid,0
4,c,12,2025/01/02,d@example.invalid,1
"""

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

# %% 8. Completeness

# Is the required value present?

print("\nStep 8 - Completeness")

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

print("Missing value IDs:", table.loc[table["value"].isna(), "id"].tolist())

# %% 9. Validity

# Does a present value obey the allowed range or format?

print("\nStep 9 - Validity")

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

present = table["value"].notna()
invalid = present & ~table["value"].between(0, 100)
print("Invalid values:", table.loc[invalid, "value"].tolist())
parsed = pd.to_datetime(["2025-01-10", "01/10/2025"], format="%Y-%m-%d", errors="coerce")
print("Invalid date format:", parsed.isna().tolist())

# %% 10. Consistency

# Are equivalent values represented the same way?

print("\nStep 10 - Consistency")

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

canonical = table["group"].str.strip().str.lower()
print("Before:", table["group"].unique().tolist())
print("After:", canonical.unique().tolist())

# %% 11. Uniqueness

# Does the identifier occur once per entity?

print("\nStep 11 - Uniqueness")

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

print("Repeated IDs:", table.loc[table.duplicated("id", keep=False), "id"].tolist())

# %% 12. Timeliness

# Is the value recent enough at the chosen date?

print("\nStep 12 - Timeliness")

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

age_days = (snapshot - table["updated"]).dt.days
print("Stale updates:", table.loc[age_days > 7, "updated"].dt.date.tolist())

# %% 13. Accuracy

# Does the value agree with a trusted source? A number can be in range and still be wrong.

print("\nStep 13 - Accuracy")

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

disagrees = present & ~invalid & table["value"].ne(table["reference"])
print(table.loc[disagrees, ["id", "value", "reference"]])

# %% 14. MCAR

# Missing completely at random means the chance of a blank is independent of both observed and missing values. We choose row positions at random.

print("\nStep 14 - MCAR")

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(42)
full = pd.DataFrame({"group": ["a", "a", "b", "b", "a", "b"],
                     "value": [2, 4, 6, 8, 10, 12]})
incomplete = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6],
                           "group": ["a", "a", "a", "b", "b", "b"],
                           "value": [2, np.nan, 6, 8, np.nan, 12]})

mcar = full["value"].copy().astype(float)
random_rows = rng.choice(len(full), size=2, replace=False)
mcar.iloc[random_rows] = np.nan
print("MCAR:", mcar.tolist())

# %% 15. MAR

# Missing at random means an observed variable explains the missingness. Here the observed group controls it.

print("\nStep 15 - MAR")

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(42)
full = pd.DataFrame({"group": ["a", "a", "b", "b", "a", "b"],
                     "value": [2, 4, 6, 8, 10, 12]})
incomplete = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6],
                           "group": ["a", "a", "a", "b", "b", "b"],
                           "value": [2, np.nan, 6, 8, np.nan, 12]})

print("MAR:", full["value"].mask(full["group"].eq("b")).tolist())

# %% 16. MNAR

# Missing not at random means the missing value itself can influence whether it is recorded.

print("\nStep 16 - MNAR")

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(42)
full = pd.DataFrame({"group": ["a", "a", "b", "b", "a", "b"],
                     "value": [2, 4, 6, 8, 10, 12]})
incomplete = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6],
                           "group": ["a", "a", "a", "b", "b", "b"],
                           "value": [2, np.nan, 6, 8, np.nan, 12]})

print("MNAR:", full["value"].mask(full["value"] > 8).tolist())

# %% 17. Drop rows or fill a simple median

# Dropping blanks changes which rows remain. A median fill leaves all rows in place and supplies one common replacement.

print("\nStep 17 - Drop rows or fill a simple median")

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(42)
full = pd.DataFrame({"group": ["a", "a", "b", "b", "a", "b"],
                     "value": [2, 4, 6, 8, 10, 12]})
incomplete = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6],
                           "group": ["a", "a", "a", "b", "b", "b"],
                           "value": [2, np.nan, 6, 8, np.nan, 12]})

print("Rows after dropping:", len(incomplete.dropna(subset=["value"])))
values = incomplete["value"]
print("Median fill:", values.fillna(values.median()).tolist())

# %% 18. Fill using groups and keep an indicator

# Group medians can differ. Record which rows had missing values before filling.

print("\nStep 18 - Fill using groups and keep an indicator")

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(42)
full = pd.DataFrame({"group": ["a", "a", "b", "b", "a", "b"],
                     "value": [2, 4, 6, 8, 10, 12]})
incomplete = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6],
                           "group": ["a", "a", "a", "b", "b", "b"],
                           "value": [2, np.nan, 6, 8, np.nan, 12]})

missing = values.isna()
group_medians = incomplete.groupby("group")["value"].transform("median")
print(incomplete.assign(value_filled=values.fillna(group_medians),
                        value_missing=missing))

# %% 19. Model-based fill

# When another feature predicts the missing column, learn the relationship from observed rows and predict the blanks. In a modelling pipeline, learn this relationship on training rows.

print("\nStep 19 - Model-based fill")

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(42)
full = pd.DataFrame({"group": ["a", "a", "b", "b", "a", "b"],
                     "value": [2, 4, 6, 8, 10, 12]})
incomplete = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6],
                           "group": ["a", "a", "a", "b", "b", "b"],
                           "value": [2, np.nan, 6, 8, np.nan, 12]})

imputer = LinearRegression().fit(incomplete.loc[~missing, ["x"]], values[~missing])
predicted = imputer.predict(incomplete.loc[missing, ["x"]])
print("Model fills:", predicted.round(2).tolist())
