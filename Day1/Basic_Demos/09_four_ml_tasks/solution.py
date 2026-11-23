"""Day 1 | Slides 9 | Regression, classification, clustering, and forecasting (complete solution)."""

# %% Setup

from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier

# %% 1. Regression

print("\nStep 1: Regression")
reg = LinearRegression().fit([[1], [2], [3]], [3, 5, 7])
print("Numeric prediction:", round(reg.predict([[4]])[0], 1))

# %% 2. Classification

print("\nStep 2: Classification")
clf = DecisionTreeClassifier(max_depth=1, random_state=0)
clf.fit([[0], [1], [8], [9]], ["low", "low", "high", "high"])
print("Predicted classes:", clf.predict([[2], [7]]).tolist())
three = DecisionTreeClassifier(max_depth=2, random_state=0)
three.fit([[0], [1], [4], [5], [8], [9]], ["low", "low", "middle", "middle", "high", "high"])
print("Three-class prediction:", three.predict([[4]])[0])

# %% 3. Clustering

print("\nStep 3: Clustering")
points = [[0], [1], [8], [9]]
clusters = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(points)
print("Group IDs:", clusters.tolist())

# %% 4. Forecasting

print("\nStep 4: Forecasting")
time_train = [[1], [2], [3]]
trend = LinearRegression().fit(time_train, [10, 12, 14])
print("Time 4 forecast:", round(trend.predict([[4]])[0], 1), "Actual:", 16)
