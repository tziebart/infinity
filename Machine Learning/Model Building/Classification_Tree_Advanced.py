import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder

# import data
data_for_model = pd.read_pickle("data/abc_classification_modelling.p")
# data_for_model = pickle.load(open("data/abc_classification_modelling.p", "rb"))
# drop customer_id - don't need it.
data_for_model.drop("customer_id", axis = 1, inplace = True)
#shuffle data
data_for_model = shuffle(data_for_model, random_state = 42)

# check class balance

data_for_model["signup_flag"].value_counts(normalize = True)

# deal with missing values
data_for_model.isna().sum()
# because so few missing values we will drop the rows.
data_for_model.dropna(how = "any", inplace = True)

# split input and output variables
X = data_for_model.drop(["signup_flag"], axis = 1)
y = data_for_model["signup_flag"]

# split training and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

# deal with categorical values
categorical_vars = ["gender"]

one_hot_encoder = OneHotEncoder(sparse_output = False, drop = "first")

X_train_encoded = one_hot_encoder.fit_transform(X_train[categorical_vars])
X_test_encoded = one_hot_encoder.transform(X_test[categorical_vars])

encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)

X_train_encoded = pd.DataFrame(X_train_encoded, columns = encoder_feature_names)
X_train = pd.concat([X_train.reset_index(drop=True), X_train_encoded.reset_index(drop=True)], axis = 1)
X_train.drop(categorical_vars, axis =1, inplace = True)

X_test_encoded = pd.DataFrame(X_test_encoded, columns = encoder_feature_names)
X_test = pd.concat([X_test.reset_index(drop=True), X_test_encoded.reset_index(drop=True)], axis = 1)
X_test.drop(categorical_vars, axis =1, inplace = True)

# Model training

clf = DecisionTreeClassifier(random_state=42, max_depth=5)
clf.fit(X_train, y_train)

# Model Assessment

y_pred_class = clf.predict(X_test)
y_pred_prob = clf.predict_proba(X_test)[:, 1]

# confusion matrix

conf_matrix = confusion_matrix(y_test, y_pred_class)

plt.style.use("seaborn-v0_8-poster")
plt.matshow(conf_matrix, cmap = "coolwarm")
plt.gca().xaxis.tick_bottom()
plt.title("Confusion Matrix")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
for (i, j), corr_value in np.ndenumerate(conf_matrix):
    plt.text(j, i, corr_value, ha = "center", va = "center", fontsize = 20)
plt.show()

# Accuracy (the number of correct classification out of all attemtped classifications)

accuracy_score(y_test, y_pred_class)

# precision (of all observations that were predicted as positive, how many were actually positive)
precision_score(y_test, y_pred_class)

# Recall (of all positive observations, how many did we predict as positive)
recall_score(y_test, y_pred_class)

# F1 Score (the harmonic mean of precision and recall)
f1_score(y_test, y_pred_class)

# finding best max_depth
max_depth_list = list(range(1,15))
accuracy_scores = []

for depth in max_depth_list:
    
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = f1_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

max_accuracy = max(accuracy_scores)
max_accuracy_idx = accuracy_scores.index(max_accuracy)
optimal_depth = max_depth_list[max_accuracy_idx]

plt.plot(max_depth_list, accuracy_scores)
plt.scatter(optimal_depth, max_accuracy, marker="x", color="red")
plt.title(f"Accuracy (F1 Score) by Max Depth \n Optimal Tree Depth: {optimal_depth} (Accuracy: {round(max_accuracy), 4}")
plt.xlabel("Max Depth of Decision Tree")
plt.ylabel("Accuracy (F1 Score")
plt.tight_layout()
plt.show()

# plot our model
plt.figure(figsize=(25,15))
tree = plot_tree(clf, 
                 feature_names = list(X.columns),
                 filled=True,
                 rounded=True,
                 fontsize=16)

