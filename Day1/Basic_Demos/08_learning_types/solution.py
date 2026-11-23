"""Day 1 | Slides 8 | Supervised, unsupervised, reinforcement, and other learning types (complete solution)."""

# %% Setup

from collections import Counter
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.semi_supervised import LabelSpreading

x = [[0], [0.2], [0.4], [5], [5.2], [5.4]]

# %% 1. Supervised

print("\nStep 1: Supervised")
model = LinearRegression().fit([[1], [2], [4]], [2, 4, 8])
print("Prediction for 3:", round(model.predict([[3]])[0], 2))

# %% 2. Unsupervised clustering

print("\nStep 2: Unsupervised clustering")
groups = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(x)
print("Group IDs:", groups.tolist())

# %% 3. Unsupervised dimension reduction

print("\nStep 3: Unsupervised dimension reduction")
component = PCA(n_components=1).fit_transform([[1, 2], [2, 4], [3, 6]])
print("Reduced shape:", component.shape)

# %% 4. Unsupervised anomaly detection

print("\nStep 4: Unsupervised anomaly detection")
points = [[1], [1.1], [1.2], [1.3], [20]]
flags = IsolationForest(contamination=0.2, random_state=0).fit_predict(points)
print("Anomaly flags:", flags.tolist(), "(-1 means unusual)")

# %% 5. Reinforcement learning

print("\nStep 5: Reinforcement learning")
q = {"A": 0.0, "B": 0.0}
for action, reward in [("A", 1), ("B", 3), ("A", 1), ("B", 3)]:
    q[action] += 0.5 * (reward - q[action])
print("Action values:", q, "Chosen action:", max(q, key=q.get))

# %% 6. Semi-supervised

print("\nStep 6: Semi-supervised")
partly_labelled = [0, -1, -1, -1, -1, 1]
spread = LabelSpreading().fit(x, partly_labelled)
print("Inferred labels:", spread.transduction_.tolist())

# %% 7. Self-supervised

print("\nStep 7: Self-supervised")
tokens = "A B A B A C".split()
pairs = list(zip(tokens, tokens[1:]))
after_a = Counter(target for source, target in pairs if source == "A")
print("Generated pairs:", pairs)
print("Most likely next token after A:", after_a.most_common(1)[0][0])
