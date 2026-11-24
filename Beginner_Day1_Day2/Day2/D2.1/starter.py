""" D2.1 - Training, validation and learning

We will compare simple predictions, see what model complexity changes and practise splitting data. This helps learners understand how to judge a model fairly.
"""

# %% Setup

import numpy as np
import pandas as pd

# %% 1. Start with a baseline

# The average training value is a simple first guess.

print("\nStep 1 - Start with a baseline")

training = np.array([2,4,6,8])
print('Average baseline:', training.mean())

# %% 2. Underfitting and overfitting

# We compare polynomial curves of different complexity using the same points and separate checks.

print("\nStep 2 - Underfitting and overfitting")

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
x = np.array([0,1,2,3,4]).reshape(-1,1)
y = np.array([0,1,4,9,16])
for degree in [1,2,4]:
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(x,y)
    print('Degree', degree, 'prediction for 5:', round(model.predict([[5]])[0],2))

# %% 3. Learning with more examples

# We train on two different training sizes and compare their predictions.

print("\nStep 3 - Learning with more examples")

from sklearn.linear_model import LinearRegression
x = np.array([1,2,3,4]).reshape(-1,1)
y = np.array([2,4,6,8])
for count in [2,4]:
    model = LinearRegression().fit(x[:count], y[:count])
    print('Training rows:', count, 'prediction:', round(model.predict([[5]])[0],1))

# %% 4. Cross-validation and hyperparameters

# We change a tree's maximum depth and evaluate each setting across several held-out folds.

print("\nStep 4 - Cross-validation and hyperparameters")

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
x = np.arange(12).reshape(-1,1)
y = np.arange(12)
for depth in [1,2]:
    model = DecisionTreeRegressor(max_depth=depth, random_state=0)
    scores = cross_val_score(model,x,y,cv=3,scoring='neg_mean_absolute_error')
    print('Depth:', depth, 'mean error:', round(-scores.mean(),2))

# %% 5. Gradient descent and learning rate

# We update a number in the direction that reduces a simple loss.

print("\nStep 5 - Gradient descent and learning rate")

weight = 0.0
for step_number in range(4):
    gradient = 2 * (weight - 3)
    weight = weight - 0.25 * gradient
    print('Step', step_number + 1, 'weight', round(weight,2))

# PRACTISE: Change one numeric input in one step and rerun that step.
# Compare with the original output. Restore the value and try another step.
