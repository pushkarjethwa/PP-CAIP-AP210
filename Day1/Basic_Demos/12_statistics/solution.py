"""Day 1 | Slides 12 | Centre, spread, skew, and four distributions (complete solution)."""

# %% Setup

import sys

import numpy as np
import pandas as pd

values = pd.Series([1, 2, 2, 3, 12])
rng = np.random.default_rng(42)

# %% 1. Mean and median

print("\nStep 1: Mean and median")
print("Mean:", values.mean(), "Median:", values.median())

# %% 2. Variance and standard deviation

print("\nStep 2: Variance and standard deviation")
print("Sample variance:", round(values.var(), 2))
print("Sample standard deviation:", round(values.std(), 2))

# %% 3. Skew and a log

print("\nStep 3: Skew and a log")
print("Skew before log:", round(values.skew(), 2))
print("Skew after log:", round(np.log(values).skew(), 2))

# %% 4. Normal, log-normal, binomial, and Poisson

print("\nStep 4: Normal, log-normal, binomial, and Poisson")
samples = {
    "Normal": rng.normal(0, 1, 500),
    "Log-normal": rng.lognormal(0, 1, 500),
    "Binomial": rng.binomial(10, 0.3, 500),
    "Poisson": rng.poisson(3, 500),
}
for name, sample in samples.items():
    print(name, "mean =", round(sample.mean(), 2),
          "range =", np.round([sample.min(), sample.max()], 2))

if "--plot" in sys.argv:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(7, 5))
    for ax, (name, sample) in zip(axes.flat, samples.items()):
        bins = np.arange(sample.max() + 2) - 0.5 if sample.dtype.kind == "i" else 24
        ax.hist(sample, bins=bins)
        ax.set(title=name, xlabel="Value", ylabel="Frequency")
    fig.tight_layout()
    plt.show()
