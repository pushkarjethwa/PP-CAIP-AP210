"""Day 1 | Slides 6 | When a rule or a model makes sense (complete solution)."""

# %% Setup

from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier

values = [2, 4, 6]
labels_by_code = {"A": "alpha", "B": "beta"}
x_train = [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]
y_train = [0, 0, 1, 1, 1, 1]
x_check = [[0, 0], [1, 1], [2, 1]]
y_check = [0, 1, 0]

# %% 1. Fixed rule and lookup

print("\nStep 1: Fixed rule and lookup")
print("Above 4:", [value > 4 for value in values])
print("Code B:", labels_by_code["B"])

# %% 2. Learn and check mistakes

print("\nStep 2: Learn and check mistakes")
model = DecisionTreeClassifier(max_depth=1, random_state=0).fit(x_train, y_train)
predicted = model.predict(x_check)
print("Predicted:", predicted.tolist(), "Actual:", y_check)
print("Confusion matrix:\n", confusion_matrix(y_check, predicted, labels=[0, 1]))
