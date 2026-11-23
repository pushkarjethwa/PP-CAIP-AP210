"""Day 1 | Slides 1–4 | Python and package check (complete solution)."""

# %% Setup

import sys

import matplotlib
import numpy as np
import pandas as pd
import pyarrow
import scipy
import sklearn

# %% 1. Check Python

print("\nStep 1: Check Python")
print("Python", sys.version.split()[0])
print("Interpreter", sys.executable)

# %% 2. Check the imports

print("\nStep 2: Check the imports")
print("numpy", np.__version__)
print("pandas", pd.__version__)
print("pyarrow", pyarrow.__version__)
print("scipy", scipy.__version__)
print("scikit-learn", sklearn.__version__)
print("matplotlib", matplotlib.__version__)
