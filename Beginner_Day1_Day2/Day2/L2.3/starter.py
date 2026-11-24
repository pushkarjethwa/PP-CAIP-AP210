""" L2.3 - Model training and tuning

We will join inputs, compare model settings and look at errors. This shows why selection and testing require separate data.
"""

# %% Setup

import numpy as np
import pandas as pd

# %% 1. Join tables and inspect missing values

# A join combines known attributes with features while making missing rows visible.

print("\nStep 1 - Join tables and inspect missing values")

left = pd.DataFrame({'id':[1,2,3], 'value':[10,20,30]})
right = pd.DataFrame({'id':[1,2], 'activity':[2,3]})
joined = left.merge(right,on='id',how='left')
print(joined); print('Missing:', joined['activity'].isna().sum())

# %% 2. Gradient descent updates

# A learning rate controls how much the weight changes each step.

print("\nStep 2 - Gradient descent updates")

weight = 0.0
for number in range(3):
    gradient = 2 * (weight - 3)
    weight = weight - 0.25 * gradient
    print('Step', number + 1, ':', round(weight,2))

# %% 3. Baseline and regularised model

# A baseline gives a simple reference for a learned line.

print("\nStep 3 - Baseline and regularised model")

from sklearn.linear_model import Ridge
x = [[1],[2],[3],[4]]
y = [2,4,6,8]
model = Ridge(alpha=1).fit(x,y)
print('Mean baseline:', np.mean(y))
print('Ridge prediction:', round(model.predict([[5]])[0],2))

# %% 4. Search for a setting

# We compare several candidate values of alpha with cross-validation.

print("\nStep 4 - Search for a setting")

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
x = np.arange(1,9).reshape(-1,1)
y = np.arange(1,9)*2
search = GridSearchCV(Ridge(), {'alpha':[0.1,1.0,10.0]}, cv=2)
search.fit(x,y)
print('Chosen setting:', search.best_params_)

# %% 5. Nested cross-validation

# An outer split provides a check while an inner process chooses settings.

print("\nStep 5 - Nested cross-validation")

from sklearn.model_selection import KFold
outer = KFold(n_splits=2)
for train, test in outer.split(np.arange(6)):
    print('Inner selection uses:', train, 'Outer check uses:', test)

# %% 6. Error and selection bias

# We compare errors across observed groups and notice missing groups.

print("\nStep 6 - Error and selection bias")

check = pd.DataFrame({'group':['A','A','B'], 'actual':[10,20,30], 'predicted':[12,19,25]})
check['error'] = (check['actual'] - check['predicted']).abs()
print(check.groupby('group')['error'].mean())

# %% 7. Conservative estimates

# A lower percentile gives a cautious value from a sample.

print("\nStep 7 - Conservative estimates")

estimates = pd.Series([10,12,14,16,18])
print('20th percentile:', estimates.quantile(0.2))

# PRACTISE: Change one numeric input in one step and rerun that step.
# Compare with the original output. Restore the value and try another step.
