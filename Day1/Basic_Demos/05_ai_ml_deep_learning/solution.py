"""Day 1 | Slides 5 | AI, ML, deep learning, and data science (complete solution)."""

# %% Setup

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier

numbers = [1, 2, 3, 4]
x = [[1], [2], [3], [4]]
y = [3, 5, 7, 9]
bits = [[0, 0], [0, 1], [1, 0], [1, 1]]
different = [0, 1, 1, 0]

# %% 1. Written rule

print("\nStep 1: Written rule")
print([value >= 3 for value in numbers])

# %% 2. Learned rule

print("\nStep 2: Learned rule")
line = LinearRegression().fit(x, y)
print("Prediction for 5:", round(line.predict([[5]])[0], 2))

# %% 3. Neural network

print("\nStep 3: Neural network")
net = MLPClassifier(hidden_layer_sizes=(8, 8), solver="lbfgs",
                    random_state=0, max_iter=1000).fit(bits, different)
print("Hidden layers:", net.hidden_layer_sizes)
print("Training predictions:", net.predict(bits).tolist())

# %% 4. Data science without a model

print("\nStep 4: Data science without a model")
values = pd.Series(numbers)
print("Count:", values.count(), "Mean:", values.mean())
