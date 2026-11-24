""" L2.1 - Regression and valuation

We will inspect values, make a baseline, train a line and compare model settings. Each result shows a small part of a valuation workflow.
"""

# %% Setup

import numpy as np
import pandas as pd

# %% 1. Inspect and flag an invalid value

# We find entries outside an allowed range before modelling.

print("\nStep 1 - Inspect and flag an invalid value")

data = pd.DataFrame({'size':[1,2,3,4,5,6], 'price':[12,20,30,40,50,-1]})
print('Invalid rows:'); print(data[data['price'] <= 0])
clean = data[data['price'] > 0]
print('Usable rows:', len(clean))

# %% 2. Split and calculate a baseline

# We use earlier rows to train and keep the last row for checking.

print("\nStep 2 - Split and calculate a baseline")

data = pd.DataFrame({'size':[1,2,3,4,5,6], 'price':[12,19,31,39,51,59]})
train, test = data.iloc[:4], data.iloc[4:]
print('Baseline:', train['price'].mean())
print('Test actual:', test['price'].tolist())

# %% 3. Simple linear regression

# We fit a line and read its slope and intercept.

print("\nStep 3 - Simple linear regression")

from sklearn.linear_model import LinearRegression
x = [[1],[2],[3],[4]]
y = [12,19,31,39]
model = LinearRegression().fit(x,y)
print('Slope:', round(model.coef_[0],2), 'Intercept:', round(model.intercept_,2))
print('Prediction:', round(model.predict([[5]])[0],2))

# %% 4. Multiple inputs and a log target

# We use two columns and show how a log compresses positive values.

print("\nStep 4 - Multiple inputs and a log target")

from sklearn.linear_model import LinearRegression
x = [[1,0],[2,1],[3,0],[4,1]]
price = np.array([12,22,30,40])
model = LinearRegression().fit(x,np.log(price))
print('Predicted value:', round(float(np.exp(model.predict([[5,1]])[0])),2))

# %% 5. Ridge and Lasso

# Regularisation controls coefficient size. We compare two methods using the same examples.

print("\nStep 5 - Ridge and Lasso")

from sklearn.linear_model import Ridge, Lasso
x = [[1],[2],[3],[4]]
y = [12,19,31,39]
for model in [Ridge(alpha=1), Lasso(alpha=1)]:
    model.fit(x,y)
    print(type(model).__name__, 'slope:', round(model.coef_[0],2))

# %% 6. Compare test errors by group

# We compare actual and predicted values for two groups.

print("\nStep 6 - Compare test errors by group")

check = pd.DataFrame({'group':['A','B'], 'actual':[50,60], 'predicted':[48,63]})
check['absolute_error'] = (check['actual'] - check['predicted']).abs()
print(check.groupby('group')['absolute_error'].mean())

# %% 7. Check later data and record the model

# We keep a later observation apart and record the model's inputs.

print("\nStep 7 - Check later data and record the model")

train_times = [1,2,3,4]
test_times = [5,6]
print('Training times:', train_times, 'Later check:', test_times)
print('Model inputs:', ['size', 'rooms'])
