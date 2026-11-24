""" D1.2 - Transformations, categories and leakage

We will change numbers and categories into model inputs and check whether any information comes from the future. This helps make model results trustworthy.
"""

# %% Setup

import numpy as np
import pandas as pd

# %% 1. Standardisation

# Subtract the training mean and divide by the training standard deviation. The fitted training values then have mean 0 and standard deviation 1.

print("\nStep 1 - Standardisation")

import sys

import numpy as np
import pandas as pd
from scipy.stats import boxcox, skew
from sklearn.preprocessing import MinMaxScaler, StandardScaler

train = np.array([1, 2, 4, 8, 32, 128], dtype=float).reshape(-1, 1)

scaler = StandardScaler().fit(train)
standard = scaler.transform(train).ravel()
print("Standardised:", standard.round(2))
print(f"Mean {standard.mean():.2f} | Population SD {standard.std():.2f}")

# %% 2. Min-max and outlier sensitivity

# The training minimum maps to 0 and the maximum maps to 1. One large training value squeezes most other values together.

print("\nStep 2 - Min-max and outlier sensitivity")

import sys

import numpy as np
import pandas as pd
from scipy.stats import boxcox, skew
from sklearn.preprocessing import MinMaxScaler, StandardScaler

train = np.array([1, 2, 4, 8, 32, 128], dtype=float).reshape(-1, 1)

minmax = MinMaxScaler().fit_transform(train).ravel()
without_outlier = MinMaxScaler().fit_transform(train[:-1]).ravel()
print("Min-max with 128:", minmax.round(2))
print("Min-max without 128:", without_outlier.round(2))

# %% 3. Log and Box-Cox

# Taking a log can compress a right tail. Box-Cox chooses a power from positive training values. For new inputs we would reuse that fitted power.

print("\nStep 3 - Log and Box-Cox")

import sys

import numpy as np
import pandas as pd
from scipy.stats import boxcox, skew
from sklearn.preprocessing import MinMaxScaler, StandardScaler

train = np.array([1, 2, 4, 8, 32, 128], dtype=float).reshape(-1, 1)

logged = np.log(train.ravel())
powered, power = boxcox(train.ravel())
for name, values in [("Raw", train.ravel()), ("Standard", standard),
                     ("Min-max", minmax), ("Log", logged), ("Box-Cox", powered)]:
    print(name, "skew:", round(skew(values), 2))
print("Fitted Box-Cox power:", round(power, 2))

# %% 4. Binning

# Choose the cut points ahead of time when fixed bands make the output easier to use.

print("\nStep 4 - Binning")

import sys

import numpy as np
import pandas as pd
from scipy.stats import boxcox, skew
from sklearn.preprocessing import MinMaxScaler, StandardScaler

train = np.array([1, 2, 4, 8, 32, 128], dtype=float).reshape(-1, 1)

bands = pd.cut(train.ravel(), bins=[0, 5, 35, np.inf],
               labels=["small", "medium", "large"], right=False)
print("Bands:", list(zip(train.ravel().tolist(), bands.tolist())))

# %% 5. Fit on training, reuse on new values

# Only the training data determines the scale. A new value can exceed the training maximum and map outside 0 to 1.

print("\nStep 5 - Fit on training, reuse on new values")

import sys

import numpy as np
import pandas as pd
from scipy.stats import boxcox, skew
from sklearn.preprocessing import MinMaxScaler, StandardScaler

train = np.array([1, 2, 4, 8, 32, 128], dtype=float).reshape(-1, 1)

print("New min-max value:", MinMaxScaler().fit(train).transform([[256]]).round(2))
print("New standard value:", scaler.transform([[256]]).round(2))
print("Unchanged training mean:", scaler.mean_)

# %% 6. See the shape

# Compare the right tail before and after scaling or logging.

print("\nStep 6 - See the shape")

import sys

import numpy as np
import pandas as pd
from scipy.stats import boxcox, skew
from sklearn.preprocessing import MinMaxScaler, StandardScaler

train = np.array([1, 2, 4, 8, 32, 128], dtype=float).reshape(-1, 1)

if "--plot" in sys.argv:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(10, 3))
    for ax, (name, values) in zip(axes, [("Raw", train.ravel()),
             ("Standard", standard), ("Min-max", minmax), ("Log", logged)]):
        ax.hist(values, bins=5)
        ax.set(title=name, xlabel="Value", ylabel="Count")
    fig.tight_layout()
    plt.show()

# %% 7. One-hot encoding

# Each category gets its own indicator column. Red, blue, and green have no rank.

print("\nStep 7 - One-hot encoding")

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder

colors = [["red"], ["red"], ["blue"], ["blue"], ["green"], ["yellow"]]
sizes = [["small"], ["medium"], ["large"]]
ids = np.array([["a"], ["b"], ["c"], ["d"], ["e"], ["f"], ["g"], ["h"]])
target = np.array([0., 1., 0., 1., 1., 0., 1., 0.])

onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(colors)
print(onehot.get_feature_names_out().tolist())
print(onehot.transform([["red"], ["blue"]]).astype(int))

# %% 8. Unknown category

# Use the encoder we already fitted. With handle_unknown set to ignore, a new category becomes all zeros.

print("\nStep 8 - Unknown category")

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder

colors = [["red"], ["red"], ["blue"], ["blue"], ["green"], ["yellow"]]
sizes = [["small"], ["medium"], ["large"]]
ids = np.array([["a"], ["b"], ["c"], ["d"], ["e"], ["f"], ["g"], ["h"]])
target = np.array([0., 1., 0., 1., 1., 0., 1., 0.])

print("Unseen purple:", onehot.transform([["purple"]]).astype(int))

# %% 9. Ordinal encoding

# Small, medium, and large have an order, so we supply it explicitly.

print("\nStep 9 - Ordinal encoding")

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder

colors = [["red"], ["red"], ["blue"], ["blue"], ["green"], ["yellow"]]
sizes = [["small"], ["medium"], ["large"]]
ids = np.array([["a"], ["b"], ["c"], ["d"], ["e"], ["f"], ["g"], ["h"]])
target = np.array([0., 1., 0., 1., 1., 0., 1., 0.])

ordinal = OrdinalEncoder(categories=[["small", "medium", "large"]]).fit(sizes)
print("Ordered size codes:", ordinal.transform(sizes).ravel().astype(int).tolist())

# %% 10. Group rare categories

# Green and yellow each appear once. We can train the encoder to put those values in one infrequent column.

print("\nStep 10 - Group rare categories")

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder

colors = [["red"], ["red"], ["blue"], ["blue"], ["green"], ["yellow"]]
sizes = [["small"], ["medium"], ["large"]]
ids = np.array([["a"], ["b"], ["c"], ["d"], ["e"], ["f"], ["g"], ["h"]])
target = np.array([0., 1., 0., 1., 1., 0., 1., 0.])

rare = OneHotEncoder(min_frequency=2, handle_unknown="infrequent_if_exist",
                     sparse_output=False).fit(colors)
print(rare.get_feature_names_out().tolist())
print(rare.transform([["green"], ["yellow"], ["purple"]]).astype(int))

# %% 11. Target encoding with cross-fitting

# A simple target encoding uses a category's average target. If each ID occurs once, doing that on the same rows copies their answers. Cross-fitting uses other training rows to encode each training fold.

print("\nStep 11 - Target encoding with cross-fitting")

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder

colors = [["red"], ["red"], ["blue"], ["blue"], ["green"], ["yellow"]]
sizes = [["small"], ["medium"], ["large"]]
ids = np.array([["a"], ["b"], ["c"], ["d"], ["e"], ["f"], ["g"], ["h"]])
target = np.array([0., 1., 0., 1., 1., 0., 1., 0.])

examples = pd.DataFrame({"id": ids.ravel(), "target": target})
naive = examples.groupby("id")["target"].transform("mean")
encoder = TargetEncoder(target_type="continuous", smooth=0.0, cv=2,
                        random_state=0)
cross_fitted = encoder.fit_transform(ids, target).ravel()
print("Naive:", naive.tolist())
print("Cross-fitted:", cross_fitted.tolist())
print("Unseen ID:", encoder.transform([["new"]]).ravel().tolist())

# %% 12. Text tokens, counts, and embeddings

# Split into tokens and count each word. An embedding gives a token a short numeric vector. These tiny vectors are supplied to show their shape. Learned embeddings get their values from training.

print("\nStep 12 - Text tokens, counts, and embeddings")

import sys

import numpy as np
import pandas as pd
from scipy.signal import spectrogram
from sklearn.feature_extraction.text import CountVectorizer

texts = ["red red blue", "blue green"]
image = np.array([[0, 0, 255], [0, 255, 255]], dtype=np.uint8)
sample_rate = 1024

tokens = texts[0].split()
counts = CountVectorizer()
matrix = counts.fit_transform(texts)
print("Tokens:", tokens)
print("Vocabulary:", counts.get_feature_names_out().tolist())
print("Word counts:\n", matrix.toarray())
embedding = {"red": [1.0, 0.0], "blue": [0.8, 0.2], "green": [0.1, 0.9]}
print("Lookup vectors:\n", np.array([embedding[token] for token in tokens]))

# %% 13. Image pixels and flattening

# A grayscale image is a grid of pixel intensities. Flattening gives a row of numbers. Keeping the grid preserves the row and column layout as separate axes.

print("\nStep 13 - Image pixels and flattening")

import sys

import numpy as np
import pandas as pd
from scipy.signal import spectrogram
from sklearn.feature_extraction.text import CountVectorizer

texts = ["red red blue", "blue green"]
image = np.array([[0, 0, 255], [0, 255, 255]], dtype=np.uint8)
sample_rate = 1024

print("Image shape:", image.shape)
print("Flattened pixels:", image.ravel().tolist())

# %% 14. Target leakage

# This feature is a copy of the label produced after the outcome. A model gets a perfect score because the answer is in the input.

print("\nStep 14 - Target leakage")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

train = pd.DataFrame({"input": [0, 0, 1, 1, 0, 0, 1, 1],
                      "label": [0, 0, 1, 1, 0, 1, 0, 1]})
test = pd.DataFrame({"input": [0, 0, 0, 1, 1, 1, 1, 0],
                     "label": [0, 0, 0, 1, 1, 0, 1, 1]})

train["outcome_copy"] = train["label"]
test["outcome_copy"] = test["label"]
for names in [["input"], ["input", "outcome_copy"]]:
    model = DecisionTreeClassifier(max_depth=1, random_state=0).fit(train[names], train["label"])
    score = roc_auc_score(test["label"], model.predict_proba(test[names])[:, 1])
    print(names, "test AUC:", round(score, 2))

# %% 15. Train-test contamination

# The mean belongs to the training set. Fitting a scaler after combining both sets lets the test set change that mean.

print("\nStep 15 - Train-test contamination")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

train = pd.DataFrame({"input": [0, 0, 1, 1, 0, 0, 1, 1],
                      "label": [0, 0, 1, 1, 0, 1, 0, 1]})
test = pd.DataFrame({"input": [0, 0, 0, 1, 1, 1, 1, 0],
                     "label": [0, 0, 0, 1, 1, 0, 1, 1]})

train_values = [[1], [2], [3]]
test_values = [[30], [30]]
honest = StandardScaler().fit(train_values)
contaminated = StandardScaler().fit(train_values + test_values)
print("Training-only mean:", honest.mean_[0])
print("Contaminated mean:", contaminated.mean_[0])

# %% 16. Temporal leakage

# When the model will predict future rows, test on later rows. A shuffled split can train on dates after the ones in the test set.

print("\nStep 16 - Temporal leakage")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

train = pd.DataFrame({"input": [0, 0, 1, 1, 0, 0, 1, 1],
                      "label": [0, 0, 1, 1, 0, 1, 0, 1]})
test = pd.DataFrame({"input": [0, 0, 0, 1, 1, 1, 1, 0],
                     "label": [0, 0, 0, 1, 1, 0, 1, 1]})

time = np.arange(1, 11)
random_train, random_test = train_test_split(time, test_size=3, random_state=0)
ordered_train, ordered_test = time[:-3], time[-3:]
print("Shuffled train:", sorted(random_train), "test:", sorted(random_test))
print("Ordered train max:", ordered_train.max(), "test min:", ordered_test.min())
