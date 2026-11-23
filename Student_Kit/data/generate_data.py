"""
Horizon Gulf Bank (HGB) - Project MIZAN
Synthetic data generator for the CAIP AIP-210 courseware.

Every dataset is 100% synthetic. Horizon Gulf Bank is a fictional bank.
No record represents a real person, customer, merchant or institution.

Usage
-----
    python generate_data.py                 # full size (~2.5M rows total)
    python generate_data.py --scale 0.2     # 20% size for low-spec laptops
    python generate_data.py --only cards    # regenerate one dataset

Datasets produced (in this folder)
----------------------------------
customers_raw.csv              250k  dirty customer master (Day 1)
loan_applications.csv          300k  booked retail loans + 12m default outcome (Days 1,3,4)
property_valuations.csv        200k  mortgage collateral valuations (Day 2)
atm_cash_daily.csv             ~150k 60 ATMs x 6.8 years of daily cash withdrawals (Days 2,4)
card_transactions.csv.gz       1.5M  card transactions with fraud labels (Days 3,4,5)
card_transactions_prod_2025Q4.csv.gz  400k  "production" traffic with drift (Day 5)
kyc_documents_64.npz           15k   64x64 grayscale onboarding-document images (Day 4)
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
SEED = 2026

NATIONALITY = ["Emirati", "Arab_Expat", "South_Asian", "Western", "Other_Asian", "African", "Other"]
NAT_P = [0.12, 0.22, 0.36, 0.08, 0.14, 0.05, 0.03]
EMIRATES = ["Dubai", "Abu Dhabi", "Sharjah", "Ajman", "Ras Al Khaimah", "Fujairah", "Umm Al Quwain"]
EMI_P = [0.40, 0.28, 0.17, 0.06, 0.05, 0.02, 0.02]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def calibrate(logit, target):
    """Shift a logit vector so that mean(sigmoid(logit)) == target (bisection)."""
    lo, hi = -15.0, 15.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if sigmoid(logit + mid).mean() > target:
            hi = mid
        else:
            lo = mid
    return logit + (lo + hi) / 2


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 1. CUSTOMERS
# ---------------------------------------------------------------------------
def make_customers(n: int, rng: np.random.Generator) -> pd.DataFrame:
    nat = rng.choice(NATIONALITY, n, p=NAT_P)
    gender = np.where(rng.random(n) < 0.68, "M", "F")
    emirate = rng.choice(EMIRATES, n, p=EMI_P)

    age_mu = pd.Series(nat).map({"Emirati": 38, "Western": 41}).fillna(36).to_numpy()
    age = np.clip(rng.normal(age_mu, 10), 21, 72).round().astype(int)

    is_emirati = nat == "Emirati"
    r = rng.random(n)
    employment = np.where(
        age >= 60, np.where(r < 0.55, "Retired", "Self_Employed"),
        np.where(is_emirati,
                 np.where(r < 0.62, "Government", np.where(r < 0.88, "Private", "Self_Employed")),
                 np.where(r < 0.10, "Government", np.where(r < 0.88, "Private", "Self_Employed"))))
    r2 = rng.random(n)
    employer_cat = np.where(employment == "Government", "Government",
                    np.where(employment == "Private",
                             np.where(r2 < 0.30, "A", np.where(r2 < 0.60, "B", np.where(r2 < 0.82, "C", "Unlisted"))),
                             "Unlisted"))

    base = pd.Series(nat).map({"Emirati": 10.05, "Western": 10.15, "Arab_Expat": 9.55,
                               "South_Asian": 9.05, "Other_Asian": 9.25, "African": 9.0, "Other": 9.3}).to_numpy()
    emp_adj = pd.Series(employer_cat).map({"Government": 0.25, "A": 0.30, "B": 0.10, "C": -0.10,
                                           "Unlisted": -0.05}).to_numpy()
    salary = np.exp(base + emp_adj + 0.012 * (age - 35) + rng.normal(0, 0.45, n))
    salary = np.clip(salary, 3500, 250000).round(-1)

    salary_transfer = rng.random(n) < np.where(employment.astype(str) == "Self_Employed", 0.15,
                                               np.where(employment.astype(str) == "Retired", 0.35, 0.68))
    tenure = np.clip(rng.gamma(1.6, 34, n), 1, 300).round().astype(int)
    has_cc = rng.random(n) < sigmoid(-0.6 + 0.00006 * salary + 0.004 * tenure)
    has_pl = rng.random(n) < sigmoid(-1.4 + 0.9 * salary_transfer)
    has_auto = rng.random(n) < sigmoid(-2.2 + 0.00002 * salary)
    has_mort = rng.random(n) < sigmoid(-4.2 + 0.00005 * salary + 0.03 * (age - 30))
    num_products = 1 + has_cc.astype(int) + has_pl + has_auto + has_mort + (rng.random(n) < 0.35)

    avg_balance = np.round(salary * np.exp(rng.normal(0.3, 0.9, n)) * (0.6 + 0.004 * tenure), 2)
    card_spend = np.where(has_cc, np.round(salary * np.clip(rng.normal(0.22, 0.10, n), 0.01, 0.9), 2), 0.0)
    logins = rng.poisson(np.clip(22 - 0.25 * (age - 30), 2, 40))
    branch = rng.poisson(np.clip(0.4 + 0.03 * (age - 30), 0.1, 3))
    complaints = rng.poisson(0.18, n)
    aecb = np.clip(rng.normal(690, 75, n) + 0.0008 * (salary - 12000) + 0.15 * tenure, 300, 900).round().astype(int)
    days_since_txn = np.clip(rng.exponential(6, n), 0, 365).round().astype(int)

    churn_logit = (-2.25 - 0.012 * tenure - 0.35 * (num_products - 2) + 0.55 * complaints
                   - 0.035 * logins - 0.9 * salary_transfer + 0.025 * days_since_txn
                   + 0.25 * (nat == "Western") - 0.000004 * avg_balance + rng.normal(0, 0.5, n))
    churn = (rng.random(n) < sigmoid(calibrate(churn_logit, 0.12))).astype(int)

    since = pd.Timestamp("2025-10-31") - pd.to_timedelta(tenure * 30.4, unit="D")
    df = pd.DataFrame({
        "customer_id": [f"HGB{1000000 + i}" for i in range(n)],
        "customer_since": since.normalize(),
        "age": age, "gender": gender, "nationality_group": nat, "emirate": emirate,
        "employment_type": employment, "employer_category": employer_cat,
        "monthly_salary_aed": salary, "salary_transfer": salary_transfer.astype(int),
        "tenure_months": tenure, "num_products": num_products,
        "has_credit_card": has_cc.astype(int), "has_personal_loan": has_pl.astype(int),
        "has_auto_loan": has_auto.astype(int), "has_mortgage": has_mort.astype(int),
        "avg_balance_aed": avg_balance, "card_spend_monthly_aed": card_spend,
        "digital_logins_30d": logins, "branch_visits_90d": branch, "complaints_12m": complaints,
        "aecb_score": aecb, "days_since_last_txn": days_since_txn, "churned_6m": churn,
    })
    # Thin-file customers (new to UAE) have no bureau score -> missing NOT at random
    thin = (df.tenure_months < 10) & (rng.random(n) < 0.6) | (rng.random(n) < 0.03)
    df.loc[thin, "aecb_score"] = np.nan
    return df


def dirty_customers(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    d = df.copy()
    n = len(d)
    # Leakage columns recorded AFTER the outcome
    close = pd.Timestamp("2025-11-01") + pd.to_timedelta(rng.integers(1, 180, n), unit="D")
    d["account_closed_date"] = np.where(d.churned_6m == 1, close.strftime("%Y-%m-%d"), "")
    d["retention_offer_made"] = np.where(d.churned_6m == 1, rng.random(n) < 0.7, rng.random(n) < 0.04).astype(int)
    # PII that must never reach a model
    d.insert(1, "email", [f"customer{i}@example.com" for i in range(n)])
    d.insert(2, "mobile", ["+9715" + str(x) for x in rng.integers(10_000_000, 99_999_999, n)])

    # Inconsistent categorical spellings
    emi_variants = {"Dubai": ["Dubai", "DXB", "dubai", "Dubai ", "DUBAI"],
                    "Abu Dhabi": ["Abu Dhabi", "AUH", "abu dhabi", "Abu-Dhabi", "AbuDhabi"],
                    "Sharjah": ["Sharjah", "SHJ", "sharjah"],
                    "Ajman": ["Ajman", "AJM"], "Ras Al Khaimah": ["Ras Al Khaimah", "RAK", "Ras al-Khaimah"],
                    "Fujairah": ["Fujairah", "FUJ"], "Umm Al Quwain": ["Umm Al Quwain", "UAQ"]}
    mask = rng.random(n) < 0.18
    d.loc[mask, "emirate"] = [rng.choice(emi_variants[e]) for e in d.loc[mask, "emirate"]]
    gmap = {"M": ["M", "Male", "male", "m"], "F": ["F", "Female", "female", "f"]}
    mask = rng.random(n) < 0.12
    d.loc[mask, "gender"] = [rng.choice(gmap[g]) for g in d.loc[mask, "gender"]]

    # Salary stored as text in some source systems, missing in others
    sal = d["monthly_salary_aed"].astype(object)
    m_txt = rng.random(n) < 0.15
    sal[m_txt] = ["AED " + f"{v:,.0f}" for v in d.loc[m_txt, "monthly_salary_aed"]]
    sal[rng.random(n) < 0.04] = np.nan
    d["monthly_salary_aed"] = sal

    # Invalid ages and negative balances
    bad_age = rng.random(n) < 0.004
    d.loc[bad_age, "age"] = rng.choice([0, -1, 150, 999, 7], bad_age.sum())
    neg = rng.random(n) < 0.002
    d.loc[neg, "avg_balance_aed"] = -d.loc[neg, "avg_balance_aed"]

    # Mixed date formats
    dates = pd.to_datetime(d["customer_since"])
    fmt = rng.choice(["iso", "dmy", "text"], n, p=[0.75, 0.15, 0.10])
    d["customer_since"] = np.select([fmt == "iso", fmt == "dmy"],
                                    [dates.dt.strftime("%Y-%m-%d"), dates.dt.strftime("%d/%m/%Y")],
                                    dates.dt.strftime("%b %d %Y"))
    # Random missingness
    for c, p in [("digital_logins_30d", 0.03), ("employer_category", 0.02), ("tenure_months", 0.01)]:
        d.loc[rng.random(n) < p, c] = np.nan

    # Duplicates: 2% exact, 1% near-duplicates (same id, re-keyed later with different casing)
    exact = d.sample(frac=0.02, random_state=1)
    near = d.sample(frac=0.01, random_state=2).copy()
    near["emirate"] = near["emirate"].astype(str).str.upper()
    near["digital_logins_30d"] = near["digital_logins_30d"].fillna(0) + 1
    d = pd.concat([d, exact, near]).sample(frac=1.0, random_state=3).reset_index(drop=True)
    return d


# ---------------------------------------------------------------------------
# 2. LOAN APPLICATIONS (booked loans)
# ---------------------------------------------------------------------------
def make_loans(n: int, cust: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    idx = rng.integers(0, len(cust), n)
    c = cust.iloc[idx].reset_index(drop=True)
    dates = pd.Timestamp("2019-01-01") + pd.to_timedelta(rng.integers(0, 2373, n), unit="D")  # to mid-2025
    product = rng.choice(["Personal_Loan", "Auto_Loan", "Credit_Card"], n, p=[0.55, 0.20, 0.25])
    salary = c.monthly_salary_aed.to_numpy() * np.exp(rng.normal(0, 0.05, n))
    salary = salary.round(-1)

    mult = np.where(product == "Personal_Loan", rng.uniform(3, 20, n),
            np.where(product == "Auto_Loan", rng.uniform(4, 15, n), rng.uniform(1, 4, n)))
    amount = np.clip(salary * mult, 5000, 4_000_000).round(-2)
    tenure = np.where(product == "Personal_Loan", rng.choice([12, 24, 36, 48], n, p=[.1, .25, .35, .3]),
              np.where(product == "Auto_Loan", rng.choice([24, 36, 48, 60], n, p=[.1, .2, .3, .4]), 0))
    aecb = c.aecb_score.to_numpy()
    aecb_f = np.where(np.isnan(aecb), 640, aecb)
    prior_dq = rng.poisson(np.clip((760 - aecb_f) / 120, 0.02, 3))
    inquiries = rng.poisson(np.clip(1.2 + (720 - aecb_f) / 150, 0.2, 5))
    existing = np.round(salary * np.clip(rng.beta(2, 6, n) * 0.55, 0, 0.45), 0)

    base_rate = np.where(product == "Personal_Loan", 0.065, np.where(product == "Auto_Loan", 0.045, 0.33))
    rate = base_rate + np.clip((720 - aecb_f) / 4000, -0.02, 0.06) + rng.normal(0, 0.005, n)
    rate = np.clip(rate, 0.03, 0.42).round(4)
    r_m = rate / 12
    emi = np.where(tenure > 0, amount * r_m / (1 - (1 + r_m) ** (-np.maximum(tenure, 1))), amount * 0.05)
    dbr = np.clip((existing + emi) / salary, 0.02, 0.5)  # CBUAE 50% DBR cap for booked loans
    months_emp = np.clip(rng.gamma(2.0, 24, n), 1, 360).round().astype(int)
    residence = rng.choice(["Rented", "Owned", "Company_Provided", "With_Family"], n, p=[.55, .15, .18, .12])
    dependents = rng.poisson(1.4, n)

    year = dates.year.to_numpy()
    covid = np.asarray((dates >= "2020-03-01") & (dates <= "2020-12-31"))
    emp = c.employment_type.to_numpy()
    ecat = c.employer_category.to_numpy()

    logit = (-3.0 + 4.2 * (dbr - 0.30) - 0.012 * (aecb_f - 690) + 0.45 * prior_dq + 0.12 * inquiries
             - 0.35 * c.salary_transfer.to_numpy() - 0.005 * months_emp
             + np.select([ecat == "Government", ecat == "A", ecat == "B", ecat == "C"], [-0.8, -0.4, 0, 0.35], 0.6)
             + 0.6 * np.isnan(aecb) + 0.35 * (product == "Credit_Card") + 0.9 * covid
             + 0.25 * (emp == "Self_Employed") - 0.15 * np.log(salary / 10000)
             + 0.15 * (residence == "Rented") + rng.normal(0, 0.6, n))
    pd_true = sigmoid(calibrate(logit, 0.075))
    default = (rng.random(n) < pd_true).astype(int)
    lgd = np.clip(rng.beta(3, 2.5, n) * np.where(product == "Auto_Loan", 0.55, 0.95), 0.05, 1)
    loss = np.where(default == 1, np.round(amount * lgd * rng.uniform(0.5, 0.95, n), 0), 0.0)
    grade = pd.cut(pd_true, [0, 0.02, 0.05, 0.10, 0.20, 1.0], labels=list("ABCDE")).astype(str)

    df = pd.DataFrame({
        "application_id": [f"LN{2019000000 + i}" for i in range(n)],
        "customer_id": c.customer_id, "application_date": dates.strftime("%Y-%m-%d"),
        "product": product, "loan_amount_aed": amount, "tenure_months": tenure, "interest_rate": rate,
        "monthly_salary_aed": salary, "age": c.age, "gender": c.gender, "nationality_group": c.nationality_group,
        "emirate": c.emirate, "employment_type": emp, "employer_category": ecat,
        "months_at_employer": months_emp, "salary_transfer": c.salary_transfer, "residence_type": residence,
        "num_dependents": dependents, "aecb_score": aecb, "num_bureau_inquiries_6m": inquiries,
        "num_prior_delinquencies_24m": prior_dq, "existing_monthly_obligations_aed": existing,
        "monthly_installment_aed": np.round(emi, 2), "debt_burden_ratio": dbr.round(4),
        "risk_grade": grade,
        # --- outcome / post-outcome columns ---
        "default_12m": default, "loss_amount_aed": loss,
        "max_days_past_due": np.where(default == 1, rng.integers(90, 365, n), rng.choice([0, 0, 0, 5, 15, 30, 45], n)),
        "collection_calls": np.where(default == 1, rng.poisson(14, n), rng.poisson(0.4, n)),
    })
    return df.sort_values("application_date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. PROPERTY VALUATIONS
# ---------------------------------------------------------------------------
COMMUNITIES = {  # name: (emirate, base AED per sqft in Jan-2018, share of villas)
    "Downtown Dubai": ("Dubai", 2100, 0.0), "Dubai Marina": ("Dubai", 1650, 0.0),
    "Palm Jumeirah": ("Dubai", 2600, 0.35), "Business Bay": ("Dubai", 1500, 0.0),
    "Jumeirah Village Circle": ("Dubai", 850, 0.15), "Jumeirah Lake Towers": ("Dubai", 1050, 0.0),
    "Arabian Ranches": ("Dubai", 1150, 0.95), "Dubai Hills Estate": ("Dubai", 1500, 0.55),
    "International City": ("Dubai", 520, 0.0), "Dubai Silicon Oasis": ("Dubai", 720, 0.1),
    "Al Barsha": ("Dubai", 980, 0.3), "Mirdif": ("Dubai", 900, 0.85), "Dubai South": ("Dubai", 650, 0.3),
    "Damac Hills": ("Dubai", 950, 0.6), "Emirates Hills": ("Dubai", 3200, 1.0),
    "Saadiyat Island": ("Abu Dhabi", 1600, 0.5), "Al Reem Island": ("Abu Dhabi", 1100, 0.0),
    "Yas Island": ("Abu Dhabi", 1250, 0.45), "Khalifa City": ("Abu Dhabi", 750, 0.8),
    "Al Raha Beach": ("Abu Dhabi", 1200, 0.2), "Masdar City": ("Abu Dhabi", 900, 0.1),
    "Al Reef": ("Abu Dhabi", 700, 0.6), "Aljada": ("Sharjah", 750, 0.25), "Al Majaz": ("Sharjah", 600, 0.0),
    "Muwaileh": ("Sharjah", 520, 0.2), "Al Nahda Sharjah": ("Sharjah", 480, 0.0),
    "Al Hamra Village": ("Ras Al Khaimah", 800, 0.5), "Mina Al Arab": ("Ras Al Khaimah", 900, 0.4),
    "Emirates City": ("Ajman", 380, 0.0), "Al Nuaimiya": ("Ajman", 350, 0.1),
}


def market_index(dates: pd.DatetimeIndex, emirate: np.ndarray) -> np.ndarray:
    t = (dates - pd.Timestamp("2018-01-01")).days.to_numpy() / 365.25
    # gentle decline 2018-2020, trough mid-2020, strong rally 2021-2024, plateau 2025
    dubai = np.where(t < 2.5, 1 - 0.07 * t, 0.825 + 0.13 * (t - 2.5))
    dubai = np.where(t > 6.5, 0.825 + 0.13 * 4 + 0.03 * (t - 6.5), dubai)
    other = 1 + 0.6 * (dubai - 1)
    return np.where(emirate == "Dubai", dubai, other)


def make_properties(n: int, rng: np.random.Generator) -> pd.DataFrame:
    names = list(COMMUNITIES)
    w = np.array([3, 4, 2, 3, 5, 3, 2, 3, 3, 2, 2, 2, 2, 2, 0.4, 2, 3, 2, 2, 1.5, 1, 1.5, 2, 1.5, 1.5, 1.5, 1, 1, 1.5, 1.2])
    comm = rng.choice(names, n, p=w / w.sum())
    emirate = np.array([COMMUNITIES[c][0] for c in comm])
    base_psf = np.array([COMMUNITIES[c][1] for c in comm])
    villa_share = np.array([COMMUNITIES[c][2] for c in comm])
    r = rng.random(n)
    ptype = np.where(r < villa_share * 0.7, "Villa", np.where(r < villa_share, "Townhouse", "Apartment"))
    beds = np.where(ptype == "Apartment", rng.choice([0, 1, 2, 3, 4], n, p=[.15, .38, .30, .14, .03]),
                    rng.choice([2, 3, 4, 5, 6], n, p=[.08, .32, .33, .19, .08]))
    size = np.where(ptype == "Apartment", 420 + beds * 520, 1300 + beds * 650) * np.exp(rng.normal(0, 0.16, n))
    size = size.round()
    dates = pd.Timestamp("2018-01-01") + pd.to_timedelta(rng.integers(0, 2860, n), unit="D")
    age = np.clip(rng.gamma(2, 4.5, n), 0, 40).round().astype(int)
    off_plan = (rng.random(n) < 0.18).astype(int)
    age = np.where(off_plan == 1, 0, age)
    floor = np.where(ptype == "Apartment", np.clip(rng.gamma(1.6, 9, n), 0, 85).round(), 0).astype(int)
    view = rng.choice(["Sea_or_Marina", "Park_or_Golf", "Community", "Road"], n, p=[.15, .2, .45, .2])
    furnished = (rng.random(n) < 0.3).astype(int)
    pool = np.where(ptype == "Apartment", 0, (rng.random(n) < 0.55).astype(int))
    parking = np.where(ptype == "Apartment", np.clip(beds // 2 + 1, 1, 3), np.clip(beds // 2 + 1, 2, 4))
    metro = np.round(np.clip(rng.gamma(2, 1.6, n), 0.1, 25), 2)
    svc = np.round(np.where(ptype == "Apartment", rng.uniform(11, 28, n), rng.uniform(3, 9, n)), 2)

    mult = (market_index(dates, emirate)
            * np.where(ptype == "Villa", 0.92, np.where(ptype == "Townhouse", 0.88, 1.0))
            * pd.Series(view).map({"Sea_or_Marina": 1.22, "Park_or_Golf": 1.10, "Community": 1.0, "Road": 0.93}).to_numpy()
            * (1 - 0.009 * age) * (1 + 0.003 * np.minimum(floor, 60)) * (1 + 0.04 * furnished)
            * (1 + 0.06 * pool) * (1 - 0.012 * np.minimum(metro, 10)) * (1 - 0.06 * off_plan))
    price = base_psf * size ** 0.97 * mult * np.exp(rng.normal(0, 0.11, n))
    # A few data-entry outliers (extra zero / missing zero)
    out = rng.random(n) < 0.0015
    price = np.where(out, price * rng.choice([10, 0.1], n), price)
    df = pd.DataFrame({
        "valuation_id": [f"VAL{800000 + i}" for i in range(n)], "valuation_date": dates.strftime("%Y-%m-%d"),
        "emirate": emirate, "community": comm, "property_type": ptype, "bedrooms": beds,
        "size_sqft": size, "building_age_years": age, "floor": floor, "view": view, "furnished": furnished,
        "private_pool": pool, "parking_spaces": parking, "distance_to_metro_km": metro,
        "service_charge_aed_sqft": svc, "off_plan": off_plan, "valuation_aed": price.round(-3),
    })
    df.loc[rng.random(n) < 0.02, "building_age_years"] = np.nan
    df.loc[rng.random(n) < 0.03, "distance_to_metro_km"] = np.nan
    return df.sort_values("valuation_date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. ATM CASH DAILY
# ---------------------------------------------------------------------------
RAMADAN = [("2019-05-06", "2019-06-03"), ("2020-04-24", "2020-05-23"), ("2021-04-13", "2021-05-12"),
           ("2022-04-02", "2022-05-01"), ("2023-03-23", "2023-04-20"), ("2024-03-11", "2024-04-09"),
           ("2025-03-01", "2025-03-29")]
EID_FITR = ["2019-06-04", "2020-05-24", "2021-05-13", "2022-05-02", "2023-04-21", "2024-04-10", "2025-03-30"]
EID_ADHA = ["2019-08-11", "2020-07-31", "2021-07-20", "2022-07-09", "2023-06-28", "2024-06-16", "2025-06-06"]


def calendar_frame(start="2019-01-01", end="2025-10-31") -> pd.DataFrame:
    d = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})
    d["is_ramadan"] = 0
    for a, b in RAMADAN:
        d.loc[d.date.between(a, b), "is_ramadan"] = 1
    hol = set()
    for e in EID_FITR:
        hol |= set(pd.date_range(pd.Timestamp(e) - pd.Timedelta(days=1), periods=4))
    for e in EID_ADHA:
        hol |= set(pd.date_range(pd.Timestamp(e) - pd.Timedelta(days=1), periods=4))
    for y in range(2019, 2026):
        hol |= {pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y}-12-02"), pd.Timestamp(f"{y}-12-03")}
    d["is_public_holiday"] = d.date.isin(hol).astype(int)
    pre_eid = set()
    for e in EID_FITR + EID_ADHA:
        pre_eid |= set(pd.date_range(pd.Timestamp(e) - pd.Timedelta(days=6), periods=6))
    d["is_pre_eid_window"] = d.date.isin(pre_eid).astype(int)
    dom = d.date.dt.day
    d["is_payday_window"] = ((dom >= 25) | (dom <= 2)).astype(int)
    regime_new = d.date >= "2022-01-01"
    dow = d.date.dt.day_of_week  # Mon=0
    d["weekend_regime"] = np.where(regime_new, "Sat-Sun", "Fri-Sat")
    d["is_weekend"] = np.where(regime_new, dow.isin([5, 6]), dow.isin([4, 5])).astype(int)
    return d


def make_atm(rng: np.random.Generator, n_atm: int = 60) -> pd.DataFrame:
    cal = calendar_frame()
    T = len(cal)
    loc_types = ["Mall", "Branch", "Petrol_Station", "Residential", "Airport", "Industrial_Area"]
    lt = rng.choice(loc_types, n_atm, p=[.22, .25, .15, .2, .05, .13])
    emi = rng.choice(EMIRATES[:5], n_atm, p=[.45, .28, .15, .07, .05])
    base = {"Mall": 210000, "Branch": 150000, "Petrol_Station": 70000, "Residential": 95000,
            "Airport": 260000, "Industrial_Area": 120000}
    t = np.arange(T) / 365.25
    dates = cal.date
    covid = np.ones(T)
    lock = ((dates >= "2020-03-24") & (dates <= "2020-06-30")).to_numpy()
    covid[lock] = 0.45
    rec = ((dates > "2020-06-30") & (dates <= "2021-06-30")).to_numpy()
    covid[rec] = 0.45 + 0.55 * (np.arange(rec.sum()) / rec.sum())
    doy = dates.dt.day_of_year.to_numpy()
    summer = 1 - 0.12 * np.exp(-((doy - 205) / 30) ** 2)  # expat summer exodus
    temp = 28 + 9 * np.sin(2 * np.pi * (doy - 110) / 365.25) + rng.normal(0, 1.5, T)
    dow = dates.dt.day_of_week.to_numpy()
    regime_new = (dates >= "2022-01-01").to_numpy()
    rows = []
    for i in range(n_atm):
        b = base[lt[i]] * np.exp(rng.normal(0, 0.25))
        trend = np.exp(-0.035 * t + rng.normal(0, 0.01))          # cash usage slowly declining
        # weekly pattern depends on location and weekend regime
        wk_old = np.array([1.0, 1.0, 1.02, 1.12, 1.05, 0.95, 0.98])   # Mon..Sun ; Thu peak (pre-weekend)
        wk_new = np.array([0.98, 1.0, 1.0, 1.03, 1.12, 1.05, 0.92])   # Fri peak
        if lt[i] == "Mall":
            wk_old = wk_old * np.array([.95, .95, .95, 1.0, 1.15, 1.15, 1.0])
            wk_new = wk_new * np.array([.95, .95, .95, 1.0, 1.05, 1.2, 1.15])
        wk = np.where(regime_new, wk_new[dow], wk_old[dow])
        payday_amp = 0.55 if lt[i] == "Industrial_Area" else 0.22
        pay = 1 + payday_amp * cal.is_payday_window.to_numpy()
        ram = 1 + (0.10 if lt[i] in ("Mall", "Residential") else 0.04) * cal.is_ramadan.to_numpy()
        pre = 1 + 0.35 * cal.is_pre_eid_window.to_numpy()
        hol = 1 - (0.25 if lt[i] == "Branch" else -0.05) * cal.is_public_holiday.to_numpy()
        air = 1 + (0.2 * np.exp(-((doy - 185) / 18) ** 2) if lt[i] == "Airport" else 0)
        mu = b * trend * covid * summer * wk * pay * ram * pre * hol * air
        y = mu * np.exp(rng.normal(0, 0.09, T))
        y = np.round(y / 100) * 100
        ntx = np.round(y / rng.uniform(480, 720) * np.exp(rng.normal(0, 0.05, T)))
        outage = rng.random(T) < 0.006
        y = y.astype(float); ntx = ntx.astype(float)
        y[outage] = np.nan; ntx[outage] = np.nan
        rows.append(pd.DataFrame({"date": dates, "atm_id": f"ATM{100 + i}", "emirate": emi[i],
                                  "location_type": lt[i], "withdrawal_aed": y, "num_withdrawals": ntx}))
    df = pd.concat(rows, ignore_index=True)
    cal2 = cal.copy(); cal2["temperature_c"] = temp.round(1)
    df = df.merge(cal2, on="date", how="left")
    df["date"] = df.date.dt.strftime("%Y-%m-%d")
    return df


# ---------------------------------------------------------------------------
# 5. CARD TRANSACTIONS (+ drifted production month)
# ---------------------------------------------------------------------------
MCC = ["Grocery", "Restaurants", "Fuel", "Fashion", "Electronics", "Travel", "Online_Gaming",
       "Gold_Jewellery", "Utilities_Telecom", "Healthcare", "Entertainment", "Money_Transfer", "Luxury_Goods"]
MCC_P = np.array([.22, .17, .10, .09, .05, .05, .025, .025, .08, .05, .06, .03, .02]); MCC_P /= MCC_P.sum()
MCC_AMT = {"Grocery": 4.6, "Restaurants": 4.4, "Fuel": 4.7, "Fashion": 5.6, "Electronics": 6.5, "Travel": 7.2,
           "Online_Gaming": 4.8, "Gold_Jewellery": 7.4, "Utilities_Telecom": 5.8, "Healthcare": 5.5,
           "Entertainment": 5.0, "Money_Transfer": 7.0, "Luxury_Goods": 8.2}
MCC_RISK = {"Grocery": .05, "Restaurants": .06, "Fuel": .05, "Fashion": .15, "Electronics": .55, "Travel": .35,
            "Online_Gaming": .7, "Gold_Jewellery": .6, "Utilities_Telecom": .05, "Healthcare": .05,
            "Entertainment": .2, "Money_Transfer": .65, "Luxury_Goods": .55}
COUNTRIES = ["AE", "SA", "IN", "GB", "US", "EG", "PK", "TR", "NG", "CN", "OTHER"]


def _channel_for(mcc, rng):
    n = len(mcc)
    ecom_p = pd.Series(mcc).map({"Electronics": .45, "Online_Gaming": .95, "Travel": .7, "Fashion": .35,
                                 "Utilities_Telecom": .6, "Entertainment": .4, "Money_Transfer": .6,
                                 "Luxury_Goods": .2}).fillna(.08).to_numpy()
    r = rng.random(n)
    return np.where(r < ecom_p, "E-Commerce", np.where(rng.random(n) < 0.6, "Contactless", "POS_Chip"))


def make_card_txns(n_target: int, cust: pd.DataFrame, rng: np.random.Generator,
                   start="2025-04-01", end="2025-10-01", drift: bool = False, id_offset: int = 0) -> pd.DataFrame:
    holders = cust[cust.has_credit_card == 1]
    span_days = (pd.Timestamp(end) - pd.Timestamp(start)).days
    n_cards = max(2000, int(n_target / (25 * span_days / 183)))   # ~25 txns per card per 6 months
    h = holders.sample(n=min(n_cards, len(holders)), random_state=int(rng.integers(1e9)), replace=False).reset_index(drop=True)
    n_cards = len(h)
    span = (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds()
    activity = rng.gamma(1.3, 1, n_cards); activity = activity / activity.sum()
    counts = rng.multinomial(n_target, activity)
    card_idx = np.repeat(np.arange(n_cards), counts)
    n = len(card_idx)
    ts = pd.Timestamp(start) + pd.to_timedelta(rng.random(n) * span, unit="s")
    # people transact in daytime: bend timestamps toward 8am-11pm
    HP = np.array([.5, .3, .2, .15, .15, .3, .8, 2, 4, 5, 5.5, 6, 6.5, 6,
                                                           5.5, 5.5, 6, 6.5, 7, 7.5, 7, 5.5, 3.5, 1.5]); hour_shift = rng.choice(np.arange(24), n, p=HP / HP.sum())
    ts = ts.normalize() + pd.to_timedelta(hour_shift, unit="h") + pd.to_timedelta(rng.integers(0, 3600, n), unit="s")

    mcc = rng.choice(MCC, n, p=MCC_P)
    channel = _channel_for(mcc, rng)
    if drift:  # new channel appears in production: Digital_Wallet (Apple/Google/Samsung-style tokenised pay)
        dw = rng.random(n) < 0.22
        channel = np.where(dw & (channel != "E-Commerce"), "Digital_Wallet", channel)
    is_atm = rng.random(n) < 0.06
    mcc = np.where(is_atm, "Cash_Withdrawal", mcc)
    channel = np.where(is_atm, "ATM", channel)
    salary = h.monthly_salary_aed.to_numpy()[card_idx]
    mu = np.array([MCC_AMT.get(m, 6.2) for m in mcc]) + 0.25 * np.log(salary / 12000)
    amount = np.exp(mu + rng.normal(0, 0.75, n))
    amount = np.where(channel == "ATM", np.round(np.exp(rng.normal(6.6, 0.6, n)) / 100) * 100 + 100, amount)
    if drift:
        amount = amount * 1.12  # price inflation + new spending behaviour
    country = np.where(rng.random(n) < 0.88, "AE", rng.choice(COUNTRIES[1:], n,
                       p=[.16, .18, .14, .1, .1, .08, .08, .03, .05, .08]))
    merchant_risk = np.clip(np.array([MCC_RISK.get(m, .3) for m in mcc]) + rng.normal(0, 0.12, n), 0, 1).round(3)
    ecom = channel == "E-Commerce"
    three_ds = np.where(ecom, (rng.random(n) < 0.82).astype(float), np.nan)
    new_device = np.where(ecom | (channel == "Digital_Wallet"), (rng.random(n) < 0.06).astype(float), np.nan)
    dist = np.where(channel == "E-Commerce", np.nan,
                    np.where(country == "AE", rng.gamma(1.4, 6, n), rng.uniform(800, 7000, n))).round(1)
    is_fraud = np.zeros(n, dtype=int)
    df = pd.DataFrame({"txn_ts": ts, "card_idx": card_idx, "amount_aed": amount, "merchant_category": mcc,
                       "channel": channel, "merchant_country": country, "merchant_risk_score": merchant_risk,
                       "is_3ds_authenticated": three_ds, "is_new_device": new_device,
                       "distance_from_home_km": dist, "is_fraud": is_fraud})

    # ---- inject fraud bursts on compromised cards ----
    comp_rate = 0.022 * span_days / 183 * (1.4 if drift else 1.0)   # compromise rate per card, scaled to the window
    comp = rng.choice(n_cards, int(n_cards * comp_rate), replace=False)
    frauds = []
    for cidx in comp:
        k = rng.integers(2, 9)
        t0 = pd.Timestamp(start) + pd.Timedelta(seconds=float(rng.random() * span * 0.97))
        gaps = np.cumsum(rng.exponential(600, k))  # minutes apart -> velocity signal
        pattern = rng.choice(["cnp", "skim", "ato", "scam"], p=[.5, .2, .3, 0.0] if not drift else [.3, .1, .2, .4])
        if pattern == "cnp":
            m = rng.choice(["Electronics", "Online_Gaming", "Luxury_Goods", "Travel", "Gold_Jewellery"], k)
            ch = np.array(["E-Commerce"] * k); ctry = rng.choice(COUNTRIES, k, p=[.35, .05, .05, .15, .2, .03, .02, .05, .05, .03, .02])
            amt = np.exp(rng.normal(7.0, 0.8, k)); ds = (rng.random(k) < 0.25).astype(float); nd = (rng.random(k) < 0.7).astype(float)
            dist_f = np.full(k, np.nan)
        elif pattern == "skim":
            m = np.array(["Cash_Withdrawal"] * k); ch = np.array(["ATM"] * k)
            ctry = rng.choice(COUNTRIES[1:], k); amt = np.round(np.exp(rng.normal(7.4, 0.3, k)) / 100) * 100
            ds = np.full(k, np.nan); nd = np.full(k, np.nan); dist_f = rng.uniform(1500, 7000, k).round(1)
        elif pattern == "ato":
            m = rng.choice(["Money_Transfer", "Gold_Jewellery", "Electronics", "Luxury_Goods"], k)
            ch = rng.choice(["E-Commerce", "Contactless"], k, p=[.7, .3]); ctry = np.where(rng.random(k) < .7, "AE", "OTHER")
            amt = np.exp(rng.normal(7.6, 0.7, k)); ds = np.where(ch == "E-Commerce", (rng.random(k) < 0.5).astype(float), np.nan)
            nd = np.where(ch == "E-Commerce", 1.0, np.nan); dist_f = np.where(ch == "E-Commerce", np.nan, rng.gamma(2, 20, k)).round(1)
        else:  # social-engineering scam: customer is tricked, txn is fully authenticated, slow & low velocity -> concept drift
            k = int(rng.integers(1, 4)); gaps = np.cumsum(rng.exponential(1.5 * 86400, k))
            m = rng.choice(["Money_Transfer", "Gold_Jewellery", "Entertainment", "Fashion"], k); ch = rng.choice(["Digital_Wallet", "E-Commerce"], k)
            ctry = np.where(rng.random(k) < .9, "AE", "OTHER"); amt = np.exp(rng.normal(6.9, 0.6, k))
            ds = np.where(ch == "E-Commerce", 1.0, np.nan); nd = np.zeros(k); dist_f = np.where(ch == "E-Commerce", np.nan, rng.gamma(1.4, 6, k)).round(1)
        frauds.append(pd.DataFrame({
            "txn_ts": t0 + pd.to_timedelta(gaps, unit="s"), "card_idx": cidx, "amount_aed": amt,
            "merchant_category": m, "channel": ch, "merchant_country": ctry,
            "merchant_risk_score": np.clip([MCC_RISK.get(x, .4) + rng.normal(0.0 if pattern == "scam" else 0.1, .1) for x in m], 0, 1).round(3),
            "is_3ds_authenticated": ds, "is_new_device": nd, "distance_from_home_km": dist_f, "is_fraud": 1}))
    df = pd.concat([df, *frauds], ignore_index=True)
    df = df[df.txn_ts < pd.Timestamp(end)]
    df["amount_aed"] = df.amount_aed.round(2)
    hh = h.iloc[df.card_idx.to_numpy()]
    df["card_id"] = "C" + hh.customer_id.str[3:].to_numpy()          # one card per customer, stable across files
    df["customer_id"] = hh.customer_id.to_numpy()
    df["customer_age"] = hh.age.to_numpy()
    df["customer_home_emirate"] = hh.emirate.to_numpy()
    df["card_tenure_months"] = np.minimum(hh.tenure_months.to_numpy(), 120)
    df["customer_monthly_salary_aed"] = hh.monthly_salary_aed.to_numpy()
    df = df.sort_values("txn_ts").reset_index(drop=True)
    df.insert(0, "txn_id", [f"T{id_offset + 10_000_000 + i}" for i in range(len(df))])
    df["txn_ts"] = df.txn_ts.dt.strftime("%Y-%m-%d %H:%M:%S")
    df = df.drop(columns="card_idx")
    if drift:
        # data-quality problems that appear in production feeds
        df.loc[rng.random(len(df)) < 0.07, "merchant_risk_score"] = np.nan       # upstream enrichment outage
        df.loc[rng.random(len(df)) < 0.01, "merchant_category"] = "UNKNOWN"
    cols = ["txn_id", "txn_ts", "card_id", "customer_id", "amount_aed", "merchant_category", "channel",
            "merchant_country", "merchant_risk_score", "is_3ds_authenticated", "is_new_device",
            "distance_from_home_km", "customer_age", "customer_home_emirate", "card_tenure_months",
            "customer_monthly_salary_aed", "is_fraud"]
    return df[cols]


# ---------------------------------------------------------------------------
# 6. KYC DOCUMENT IMAGES (CNN)
# ---------------------------------------------------------------------------
DOC_CLASSES = ["cheque", "id_card", "bank_statement", "salary_certificate", "invoice"]


def make_documents(per_class: int, rng: np.random.Generator):
    from PIL import Image, ImageDraw, ImageFilter
    S = 64
    X = np.zeros((per_class * len(DOC_CLASSES), S, S), dtype=np.uint8)
    y = np.zeros(len(X), dtype=np.int64)
    k = 0
    for cls, name in enumerate(DOC_CLASSES):
        for _ in range(per_class):
            W = 128
            bg = int(rng.integers(200, 256))
            im = Image.new("L", (W, W), bg)
            d = ImageDraw.Draw(im)
            ink = int(rng.integers(0, 90))
            j = lambda a: int(a + rng.integers(-4, 5))
            if name == "cheque":
                d.rectangle([j(6), j(30), j(122), j(98)], outline=ink, width=2)
                for yy in (52, 66, 80):
                    d.line([j(14), yy + int(rng.integers(-2, 3)), j(110), yy], fill=ink, width=1)
                d.rectangle([j(86), j(36), j(116), j(46)], outline=ink)            # amount box
                d.text((j(14), j(86)), "|:" + "".join(rng.choice(list("0123456789"), 10)) + ":|", fill=ink)  # MICR
            elif name == "id_card":
                x0, y0 = j(16), j(30)
                d.rounded_rectangle([x0, y0, x0 + 96, y0 + 62], radius=8, outline=ink, width=2)
                d.rectangle([x0 + 6, y0 + 10, x0 + 32, y0 + 46], fill=ink // 2 + 60)  # photo
                for i in range(4):
                    d.line([x0 + 40, y0 + 14 + i * 10, x0 + 40 + int(rng.integers(25, 50)), y0 + 14 + i * 10], fill=ink, width=2)
                d.rectangle([x0 + 6, y0 + 52, x0 + 90, y0 + 56], fill=ink)          # barcode strip
            elif name == "bank_statement":
                d.rectangle([j(10), j(8), j(60), j(20)], fill=ink)                  # logo/header
                top = j(30)
                for r_ in range(int(rng.integers(7, 11))):
                    yy = top + r_ * 9
                    d.line([10, yy, 118, yy], fill=ink + 60, width=1)
                    for xx in (12, 52, 92):
                        d.line([xx, yy + 4, xx + int(rng.integers(12, 24)), yy + 4], fill=ink, width=1)
                for xx in (48, 88):
                    d.line([xx, top, xx, top + 90], fill=ink + 60)
            elif name == "salary_certificate":
                d.ellipse([j(10), j(8), j(34), j(32)], outline=ink, width=2)        # company seal
                for r_ in range(int(rng.integers(6, 10))):
                    yy = 44 + r_ * 8
                    d.line([j(14), yy, int(rng.integers(80, 116)), yy], fill=ink, width=2)
                d.ellipse([j(80), j(98), j(112), j(122)], outline=ink, width=2)     # stamp
                d.line([j(20), j(112), j(60), j(108)], fill=ink, width=2)          # signature
            else:  # invoice
                d.text((j(80), j(8)), "INV", fill=ink)
                d.rectangle([j(10), j(10), j(40), j(24)], outline=ink, width=2)
                top = j(36)
                d.rectangle([10, top, 118, top + 56], outline=ink, width=1)
                for r_ in range(1, 6):
                    d.line([10, top + r_ * 9, 118, top + r_ * 9], fill=ink + 70)
                d.rectangle([j(72), j(98), j(118), j(114)], outline=ink, width=3)   # totals box
            # --- realistic scan/photo variation: rotation, scale, off-centre placement, clutter, smudges ---
            im = im.rotate(float(rng.normal(0, 9)), fillcolor=bg, resample=Image.BILINEAR)
            sc = float(rng.uniform(0.55, 1.0))
            small = im.resize((int(W * sc), int(W * sc)), Image.BILINEAR)
            canvas = Image.new("L", (W, W), int(np.clip(bg + rng.integers(-25, 10), 150, 255)))
            ox, oy = int(rng.integers(0, W - small.size[0] + 1)), int(rng.integers(0, W - small.size[1] + 1))
            canvas.paste(small, (ox, oy))
            im = canvas
            d = ImageDraw.Draw(im)
            for _ in range(int(rng.integers(0, 4))):          # background clutter (desk edges, other papers)
                x0, y0 = rng.integers(0, W, 2)
                d.line([int(x0), int(y0), int(x0 + rng.integers(-60, 60)), int(y0 + rng.integers(-60, 60))],
                       fill=int(rng.integers(60, 200)), width=int(rng.integers(1, 3)))
            if rng.random() < 0.35:                             # smudge / finger / stamp occlusion
                cx, cy, r = int(rng.integers(10, W - 10)), int(rng.integers(10, W - 10)), int(rng.integers(6, 22))
                d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=int(rng.integers(90, 190)))
            if rng.random() < 0.5:
                im = im.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.3, 1.2))))
            im = im.resize((S, S), Image.BILINEAR)
            a = np.asarray(im, dtype=np.float32)
            a = a * rng.uniform(0.8, 1.1) + rng.normal(0, rng.uniform(4, 14), a.shape)
            X[k] = np.clip(a, 0, 255).astype(np.uint8)
            y[k] = cls
            k += 1
    p = rng.permutation(len(X))
    return X[p], y[p]


# ---------------------------------------------------------------------------
# Customer complaints (text) — for NLP triage, entity extraction and ASR error analysis (Demo D1.3)
COMPLAINT_CATS = ["card_dispute_fraud", "fees_charges", "loans_finance", "digital_banking", "account_kyc", "service_branch"]
_MERCH = ["an online marketplace", "a supermarket in Al Barsha", "a food delivery app", "a ride-hailing app", "an online gaming site",
          "a hotel in Istanbul", "another bank's ATM", "a merchant in Lagos", "an airport duty-free shop", "a fashion website",
          "a petrol station in Sharjah", "an electronics store in Deira"]
_ISSUES = {
    "card_dispute_fraud": [
        "I see a transaction of AED {amt} at {merch} on {date} that I did not make",
        "my credit card ending {last4} was charged AED {amt} twice by {merch}",
        "there is an unauthorised payment on my card, I never shared my OTP with anyone",
        "I raised a chargeback for AED {amt} on {date} and still no refund after {days} days",
        "my card was used abroad while it was in my wallet in Dubai",
        "someone called pretending to be the bank and now AED {amt} is missing from my card",
        "the merchant {merch} refunded AED {amt} but it never reached my card ending {last4}",
        "I want to dispute a suspicious online purchase of AED {amt}",
    ],
    "fees_charges": [
        "you charged me an annual fee of AED {amt} although it was promised free for life",
        "a late payment fee of AED {amt} was applied even though I paid on {date}",
        "why was I charged a minimum balance fee when my salary is transferred to you",
        "the foreign currency markup on my statement is much higher than what was advertised",
        "I was charged AED {amt} for an overdraft I never requested",
        "please reverse the AED {amt} fee for the cash withdrawal",
        "there is an interest charge of AED {amt} on my credit card even though I paid the full balance",
        "a service charge of AED {amt} appeared on my account without any notice",
    ],
    "loans_finance": [
        "my personal loan application was rejected with no reason given",
        "the early settlement fee you quoted for my auto loan is too high",
        "my loan instalment of AED {amt} was deducted twice this month",
        "I asked for a deferment of my EMI because of job loss and nobody replied in {days} days",
        "the interest rate on my mortgage increased without informing me",
        "I need the liability letter for my loan to close it, it is taking {days} days",
        "you are asking me to transfer my salary before approving the loan top-up",
        "the bank calculated my debt burden ratio wrongly and declined my finance",
    ],
    "digital_banking": [
        "the mobile app crashes every time I try to log in",
        "I am not receiving the OTP SMS so I cannot complete any transfer",
        "my transfer of AED {amt} to a beneficiary IBAN failed but the money left my account",
        "UAE PASS login on the app is not working since the last update",
        "the app shows the wrong balance on my savings account",
        "I cannot add a new beneficiary, the app keeps showing an error",
        "online banking locked me out after one wrong password",
        "the instant transfer took {days} days to arrive",
    ],
    "account_kyc": [
        "my account was frozen because my Emirates ID expired but I already uploaded the new one",
        "I updated my KYC documents at the branch and the account is still blocked",
        "I asked to close my account {days} days ago and it is still open",
        "please send me the bank statement for the last six months, I requested it on {date}",
        "my salary was not credited because the account is under review",
        "the bank asked for my salary certificate again even though I submitted it",
        "I cannot open a joint account, the branch says the documents are incomplete",
        "my passport details are wrong on the account and I cannot correct them",
    ],
    "service_branch": [
        "I waited more than one hour at the branch and nobody helped me",
        "the call centre kept me on hold for {days} minutes and then disconnected",
        "the staff at the branch were rude when I asked about my request",
        "the ATM did not dispense AED {amt} but my account was debited",
        "the ATM swallowed my card and the branch cannot return it",
        "I was promised a call back on {date} and nobody called",
        "the relationship manager is not answering my emails",
        "I was given wrong information by the call centre agent",
    ],
}
_OPEN = {"email": ["Dear Sir/Madam,", "Hello,", "To whom it may concern,", "Dear HGB team,", "Hi,"],
         "app_chat": ["hi", "hello", "salam", "hey", ""],
         "call_transcript": ["hello yes", "hi good morning", "salam alaikum", "yes hello", ""]}
_CLOSE = {"email": ["Please resolve this urgently.", "I expect a response within 2 working days.", "Regards,", "Thank you.",
                    "Kindly look into this."],
          "app_chat": ["pls fix", "thanks", "waiting", "plz help", "??"],
          "call_transcript": ["thank you", "ok bye", "please help me", "that is all", ""]}
# Escalation = regulator/legal threat, financial hardship or customer vulnerability (assembled from parts so wording varies)
_ESC_LEAD = ["", "Honestly ", "Please understand ", "Note that ", "I must say ", "To be clear "]
_ESC_CORE = ["I will complain to the Central Bank", "I am escalating this to the regulator", "I will file a complaint with the CBUAE",
             "my lawyer will contact you", "I am going to take legal action", "I will go to court over this",
             "I lost my job and cannot afford this", "I have no income this month and this charge leaves me with nothing",
             "my family depends on this money and I cannot pay rent", "I am a senior citizen and I do not understand these charges",
             "I am a person of determination and could not access the branch", "I am in hospital and cannot visit a branch",
             "this is my fourth complaint and nobody is taking responsibility", "I want this escalated to senior management today"]
_ESC_TAIL = [".", " and I expect an answer.", ", this is not acceptable.", ". I am very upset.", "!"]
# Hard negatives: sound similar but are NOT escalations
_NEAR_MISS = ["I read the Central Bank news about interest rates.", "My manager at work needs the salary certificate.",
              "My friend who is a lawyer recommended your bank.", "I am not complaining about the staff, they were kind.",
              "The court documents are attached for my address proof.", "I changed jobs last month so please update my employer."]
_ASR_SUB = {"fee": "free", "card": "cart", "loan": "lone", "charged": "charge", "statement": "state meant", "otp": "o t p",
            "unauthorised": "an authorized", "aed": "a e d", "iban": "i ban", "emi": "e m i", "kyc": "k y c", "atm": "a t m",
            "frozen": "frozen", "chargeback": "charge back", "refund": "re fund", "account": "a count", "transfer": "transfers",
            "branch": "brunch", "salary": "celery", "mortgage": "more gauge", "app": "up", "deducted": "the ducted"}


def _fill(t, rng):
    return t.format(amt=f"{float(np.round(rng.lognormal(6.3, 1.0), -1)):,.0f}", merch=rng.choice(_MERCH),
                    date=(pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(rng.integers(0, 640)))).strftime("%d/%m/%Y"),
                    last4=f"{int(rng.integers(0, 10000)):04d}", days=int(rng.integers(3, 60)))


def _pii(rng):
    k = rng.integers(0, 4)
    if k == 0:
        return f"My mobile is +9715{int(rng.integers(0, 10))}{int(rng.integers(1000000, 9999999))}."
    if k == 1:
        return f"Contact me on customer{int(rng.integers(1000, 99999))}@example.com."
    if k == 2:
        return f"My card number is 4{int(rng.integers(100, 999))} {int(rng.integers(1000, 9999))} {int(rng.integers(1000, 9999))} {int(rng.integers(1000, 9999))}."
    return f"My IBAN is AE{int(rng.integers(10, 99))}0{int(rng.integers(10, 99))}{int(rng.integers(10**15, 10**16 - 1))}."


def _asr(text, rng):
    words = re.sub(r"[^\w\s]", " ", text.lower()).split()
    out = []
    for w in words:
        r = rng.random()
        if r < 0.05:
            continue                                        # deletion
        if w in _ASR_SUB and rng.random() < 0.45:
            out.extend(_ASR_SUB[w].split())                # acoustic confusion
        elif r < 0.09 and len(w) > 3:
            out.append(w[: max(2, len(w) - 2)])             # truncated word
        else:
            out.append(w)
        if rng.random() < 0.04:
            out.append(rng.choice(["uh", "um", "like", "you know"]))
    return " ".join(out)


def make_complaints(n: int, cust: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    cat_p = np.array([.24, .19, .15, .18, .12, .12])
    ch_p = np.array([.40, .35, .25])
    channels = ["email", "app_chat", "call_transcript"]
    ts = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 640 * 24 * 3600, n), unit="s")
    rows = []
    for i in range(n):
        cat = rng.choice(COMPLAINT_CATS, p=cat_p)
        ch = rng.choice(channels, p=ch_p)
        parts = [rng.choice(_OPEN[ch]), _fill(rng.choice(_ISSUES[cat]), rng) + "."]
        if rng.random() < 0.35:                               # a second sentence on the same topic
            parts.append(_fill(rng.choice(_ISSUES[cat]), rng).capitalize() + ".")
        if rng.random() < 0.18:                               # cross-topic noise makes triage realistic
            other = rng.choice([c for c in COMPLAINT_CATS if c != cat])
            parts.append("Also " + _fill(rng.choice(_ISSUES[other]), rng) + ".")
        esc = rng.random() < 0.065
        if esc:
            parts.append(rng.choice(_ESC_LEAD) + rng.choice(_ESC_CORE) + rng.choice(_ESC_TAIL))
        elif rng.random() < 0.06:
            parts.append(rng.choice(_NEAR_MISS))
        if rng.random() < 0.008:                               # reviewers disagree on borderline cases
            esc = not esc
        if ch != "call_transcript" and rng.random() < 0.22:
            parts.append(_pii(rng))
        parts.append(rng.choice(_CLOSE[ch]))
        text = " ".join(p for p in parts if p)
        if ch == "app_chat":
            text = text.lower().replace("please", rng.choice(["please", "pls", "plz"])).replace("account", rng.choice(["account", "acc", "a/c"]))
        ref = None
        if ch == "call_transcript":
            ref = re.sub(r"[^\w\s]", " ", text.lower())
            ref = re.sub(r"\s+", " ", ref).strip()
            text = _asr(text, rng)
        label = cat if rng.random() > 0.04 else rng.choice(COMPLAINT_CATS)   # ~4% label noise (human tagging errors)
        rows.append((f"CMP{700000 + i}", ts[i], cust.customer_id.iloc[int(rng.integers(0, len(cust)))], ch, text, label, int(esc),
                     ref if (ref is not None and rng.random() < 0.3) else None))
    df = pd.DataFrame(rows, columns=["complaint_id", "received_ts", "customer_id", "channel", "text", "category", "escalate",
                                     "reference_transcript"])
    return df.sort_values("received_ts").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Card-linked merchant offers (implicit feedback) — for recommender systems (Demo D1.3). All merchant names are fictional.
OFFER_CATS = {  # category: (taste weights over [foodie, family, traveller, tech, commuter, luxury], fictional merchants)
    "Dining": ([1, .3, .3, .2, .2, .5], ["Marina Grill House", "Saffron Courtyard", "Corniche Shawarma Co", "The Pearl Bistro", "Al Fanar Kitchen"]),
    "Food_Delivery": ([1, .5, 0, .6, .2, 0], ["FastPlate", "DeliverEat", "Midnight Biryani", "GreenBowl Delivery"]),
    "Groceries": ([.2, 1, 0, .1, .3, .1], ["Souk Fresh Market", "Oasis Hypermarket", "Barsha Organic Grocer", "Family Mart UAE"]),
    "Kids_Family": ([0, 1, .2, 0, 0, .1], ["Funland Play Park", "KidzCity Toys", "Little Scholars Books"]),
    "Travel": ([.2, .2, 1, .1, 0, .6], ["SkyGulf Holidays", "Desert Rose Hotels", "Arabian Nights Resort", "Gulfway Travel"]),
    "Airport_Lounges": ([0, 0, 1, .2, 0, .5], ["TerminalOne Lounge", "Horizon Lounge Access"]),
    "Electronics": ([.1, .2, .1, 1, .1, .3], ["CircuitCity Gulf", "GadgetSouk", "PixelPoint Electronics"]),
    "Gaming_Streaming": ([.2, .1, 0, 1, 0, 0], ["PlayArena Online", "StreamNest", "QuestPass Gaming"]),
    "Fuel_Auto": ([0, .3, 0, .1, 1, .1], ["DuneFuel Stations", "QuickLube Auto Care", "Emirates Road Tyres"]),
    "Ride_Hailing": ([.2, 0, .3, .3, .8, 0], ["CityRide", "HopCab"]),
    "Fashion_Luxury": ([.1, .1, .3, .1, 0, 1], ["Maison Doré Boutique", "Atelier Gulf", "Gold Souk Jewellers", "Velvet Abaya House"]),
    "Wellness_Fitness": ([.1, .2, .2, .3, .1, .6], ["PulseFit Gym", "Serenity Spa", "Yoga Dunes Studio"]),
}


def make_offer_redemptions(cust: pd.DataFrame, rng: np.random.Generator, n_cust: int = 40_000) -> pd.DataFrame:
    pairs = [(c, m, w) for c, (w, ms) in OFFER_CATS.items() for m in ms]
    offers = [(f"OF{100 + i:03d}", m, c, np.array(w, float)) for i, (c, m, w) in enumerate(pairs)]
    W = np.vstack([o[3] for o in offers]) + rng.normal(0, .08, (len(offers), 6))      # offer x taste
    pop = rng.normal(0, .6, len(offers))                                              # some offers are simply more attractive
    c = cust[cust.has_credit_card == 1].sample(n_cust, random_state=int(rng.integers(1e9)))
    age, sal = c.age.to_numpy(), c.monthly_salary_aed.to_numpy()
    # persona tastes depend (noisily) on age and salary
    base = np.column_stack([np.where(age < 35, 1.6, .8), np.where((age >= 30) & (age < 50), 1.6, .6), np.where(sal > 25000, 1.6, .7),
                            np.where(age < 32, 1.5, .6), np.ones(len(c)), np.where(sal > 40000, 1.8, .4)])
    taste = rng.gamma(base * 0.35, 1.0)                                               # small shape -> 1-2 dominant tastes
    taste /= taste.sum(1, keepdims=True)
    util = 7.0 * taste @ W.T + 0.5 * pop                                              # customer x offer utility
    prob = np.exp(util); prob /= prob.sum(1, keepdims=True)
    n_red = rng.poisson(5.5, len(c)) + 1
    start, span = pd.Timestamp("2025-01-01"), 273 * 24 * 3600                        # Jan-Sep 2025
    rows = []
    for i, (cid, k) in enumerate(zip(c.customer_id.to_numpy(), n_red)):
        picks = rng.choice(len(offers), size=k, p=prob[i])
        ts = start + pd.to_timedelta(rng.integers(0, span, k), unit="s")
        for o, t in zip(picks, ts):
            rows.append((cid, offers[o][0], offers[o][1], offers[o][2], t))
    df = pd.DataFrame(rows, columns=["customer_id", "offer_id", "merchant", "offer_category", "redeemed_ts"])
    return df.sort_values("redeemed_ts").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Phone-banking (IVR) audio — synthetic 8 kHz telephone speech commands + a DTMF key-press clip (Demo D1.3)
VOICE_CMDS = ["balance", "block_card", "loan", "agent"]
_FORMANTS = {"i": (270, 2290, 3010), "e": (530, 1840, 2480), "ae": (660, 1720, 2410), "a": (730, 1090, 2440),
             "o": (570, 840, 2410), "u": (300, 870, 2240), "er": (490, 1350, 1690)}   # Peterson & Barney (1952) male averages, Hz
_WORDS = {"balance": [("burst", .07), ("ae", .30), ("er", .18), ("e", .30), ("hiss", .15)],
          "block_card": [("burst", .07), ("o", .30), ("burst", .08), ("a", .45), ("burst", .10)],
          "loan": [("o", .45), ("u", .40), ("e", .15)],
          "agent": [("e", .40), ("hiss", .12), ("i", .35), ("burst", .13)]}
DTMF = {"1": (697, 1209), "2": (697, 1336), "3": (697, 1477), "4": (770, 1209), "5": (770, 1336), "6": (770, 1477),
        "7": (852, 1209), "8": (852, 1336), "9": (852, 1477), "*": (941, 1209), "0": (941, 1336), "#": (941, 1477)}  # ITU-T Q.23


def _resonate(x, f, bw, sr):
    from scipy.signal import lfilter
    r = np.exp(-np.pi * bw / sr)
    a = [1.0, -2 * r * np.cos(2 * np.pi * f / sr), r * r]
    return lfilter([1 - r], a, x)


def _synth_word(word, sr, rng):
    from scipy.signal import butter, sosfilt
    female = rng.random() < 0.45
    f0 = rng.uniform(165, 250) if female else rng.uniform(85, 150)
    scale = rng.uniform(1.10, 1.22) if female else rng.uniform(0.94, 1.06)
    total = rng.uniform(0.42, 0.66)
    out = []
    for seg, share in _WORDS[word]:
        n = int(sr * total * share * rng.uniform(0.85, 1.15))
        if seg in ("burst", "hiss"):
            noise = rng.normal(0, 1, n)
            band = (2000, 3600) if seg == "hiss" else (1200, 3000)
            sos = butter(4, [band[0] / (sr / 2), min(band[1], 3900) / (sr / 2)], btype="band", output="sos")
            out.append(sosfilt(sos, noise) * (0.25 if seg == "hiss" else 0.5) * np.hanning(n))
            continue
        t = np.arange(n) / sr
        f0_t = f0 * (1 - 0.15 * t / max(t[-1], 1e-3))                    # falling intonation
        phase = 2 * np.pi * np.cumsum(f0_t * (1 + rng.normal(0, .01, n))) / sr
        src = (np.mod(phase, 2 * np.pi) < 0.6).astype(float)            # glottal pulse train
        y = src
        for k, F in enumerate(_FORMANTS[seg]):
            y = _resonate(y, F * scale * rng.uniform(.95, 1.05), (60, 90, 120)[k], sr)
        y = y / (np.abs(y).max() + 1e-9) * np.hanning(n) ** 0.3
        out.append(y)
    return np.concatenate(out)


def make_ivr_audio(per_class: int, rng: np.random.Generator, sr: int = 8000, clip_s: float = 0.8):
    from scipy.signal import butter, sosfilt
    tel = butter(4, [300 / (sr / 2), 3400 / (sr / 2)], btype="band", output="sos")   # telephone band
    L = int(sr * clip_s)
    X, y, snr = np.zeros((per_class * len(VOICE_CMDS), L), dtype=np.int16), [], []
    k = 0
    for c, w in enumerate(VOICE_CMDS):
        for _ in range(per_class):
            s = _synth_word(w, sr, rng)[:L]
            clip = np.zeros(L)
            off = int(rng.integers(0, L - len(s) + 1))
            clip[off:off + len(s)] = s
            db = rng.uniform(0, 25)
            p = np.mean(s ** 2)
            clip = clip + rng.normal(0, np.sqrt(p / 10 ** (db / 10)), L)
            clip = sosfilt(tel, clip)
            X[k] = np.clip(clip / (np.abs(clip).max() + 1e-9) * 0.8 * 32767, -32768, 32767).astype(np.int16)
            y.append(c); snr.append(db); k += 1
    p = rng.permutation(len(X))
    # DTMF clip: a customer keys in a menu path in the IVR
    keys = "4" + "".join(rng.choice(list("0123456789"), 5)) + "#"
    parts = []
    for ch in keys:
        tt = np.arange(int(sr * rng.uniform(.08, .14))) / sr
        lo, hi = DTMF[ch]
        parts += [0.5 * np.sin(2 * np.pi * lo * tt) + 0.5 * np.sin(2 * np.pi * hi * tt), np.zeros(int(sr * rng.uniform(.06, .12)))]
    d = np.concatenate([np.zeros(int(sr * .2))] + parts)
    d = d + rng.normal(0, .05, len(d))
    dtmf = np.clip(d * 0.6 * 32767, -32768, 32767).astype(np.int16)
    return X[p], np.array(y)[p], np.array(snr)[p].round(1), dtmf, keys


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--only", default="all", help="all|customers|loans|property|atm|cards|docs|complaints|offers|audio")
    a = ap.parse_args()
    s = a.scale
    rng = np.random.default_rng(SEED)
    t0 = time.time()

    log("customers ...")
    cust = make_customers(int(250_000 * s), rng)
    if a.only in ("all", "customers"):
        dirty_customers(cust, rng).to_csv(OUT / "customers_raw.csv", index=False)
    if a.only in ("all", "loans"):
        log("loan applications ...")
        make_loans(int(300_000 * s), cust, np.random.default_rng(SEED + 1)).to_csv(OUT / "loan_applications.csv", index=False)
    if a.only in ("all", "property"):
        log("property valuations ...")
        make_properties(int(200_000 * s), np.random.default_rng(SEED + 2)).to_csv(OUT / "property_valuations.csv", index=False)
    if a.only in ("all", "atm"):
        log("ATM cash daily ...")
        make_atm(np.random.default_rng(SEED + 3)).to_csv(OUT / "atm_cash_daily.csv", index=False)
    if a.only in ("all", "cards"):
        log("card transactions (train) ...")
        make_card_txns(int(1_500_000 * s), cust, np.random.default_rng(SEED + 4)).to_csv(
            OUT / "card_transactions.csv.gz", index=False, compression="gzip")
        log("card transactions (production Q4 with drift) ...")
        make_card_txns(int(400_000 * s), cust, np.random.default_rng(SEED + 5), start="2025-10-01",
                       end="2025-12-01", drift=True, id_offset=5_000_000).to_csv(
            OUT / "card_transactions_prod_2025Q4.csv.gz", index=False, compression="gzip")
    if a.only in ("all", "docs"):
        log("KYC document images ...")
        X, y = make_documents(int(3000 * max(s, 0.34)), np.random.default_rng(SEED + 6))
        np.savez_compressed(OUT / "kyc_documents_64.npz", X=X, y=y, classes=np.array(DOC_CLASSES))
    if a.only in ("all", "complaints"):
        log("customer complaints (text) ...")
        make_complaints(int(12_000 * max(s, 0.5)), cust, np.random.default_rng(SEED + 7)).to_csv(OUT / "customer_complaints.csv", index=False)
    if a.only in ("all", "offers"):
        log("card-linked offer redemptions ...")
        make_offer_redemptions(cust, np.random.default_rng(SEED + 9), n_cust=int(40_000 * max(s, 0.5))).to_csv(
            OUT / "card_offer_redemptions.csv", index=False)
    if a.only in ("all", "audio"):
        log("IVR audio clips ...")
        Xa, ya, snr, dtmf, keys = make_ivr_audio(150, np.random.default_rng(SEED + 8))
        np.savez_compressed(OUT / "ivr_audio_8k.npz", X=Xa, y=ya, classes=np.array(VOICE_CMDS), snr_db=snr, sr=np.array(8000),
                            dtmf=dtmf, dtmf_keys=np.array(keys))
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
