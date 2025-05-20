import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# import data

data_for_model = pd.read_csv("data/sample_data_pca.csv")

# drop user_id - don't need it.

data_for_model.drop("user_id", axis = 1, inplace = True)

#shuffle data

data_for_model = shuffle(data_for_model, random_state = 42)

# check class balance

data_for_model["purchased_album"].value_counts(normalize = True)

# deal with missing values

data_for_model.isna().sum()

# because so few missing values we will drop the rows.

data_for_model.dropna(how = "any", inplace = True)

# split input and output variables

X = data_for_model.drop(["purchased_album"], axis = 1)
y = data_for_model["purchased_album"]

# split training and test sets.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

# Feature scaling

scale_standard = StandardScaler()

X_train = scale_standard.fit_transform(X_train)
X_test = scale_standard.transform(X_test)

# Apply PCA

# Instantiate and Fit

pca = PCA(n_components= None, random_state= 42)
pca.fit(X_train)

# Extract the explained variance across components
explained_variance = pca.explained_variance_ratio_
explained_variance_cumulative = pca.explained_variance_ratio_.cumsum()

# Plot the explained variance across components
# create list for number of components

num_vars_list = list(range(1,101))
plt.figure(figsize=(15, 10))

# plot the variance explained by each component

plt.subplot(2,1,1)
plt.bar(num_vars_list, explained_variance)
plt.title("Variance across Principal Components")
plt.xlabel("Number of Components")
plt.ylabel("% Variance")
plt.tight_layout()

# plot the cumulative variance
plt.subplot(2,1,2)
plt.bar(num_vars_list, explained_variance_cumulative)
plt.title("Cumulative Variance across Principal Components")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative % Variance")
plt.tight_layout()
plt.show()

# Apply PCA with selected number of components

pca = PCA(n_components= 0.75, random_state= 42)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

pca.n_components_

# apply PCA with selected number of components

clf = RandomForestClassifier(random_state= 42)
clf.fit(X_train, y_train)

# Assess Model Accuracy

y_pred_class = clf.predict(X_test)
accuracy_score(y_test, y_pred_class)
