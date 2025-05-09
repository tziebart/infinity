
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# import sample data
my_df = pd.read_csv("data/sample_data_classification.csv")

# split into input and output objects.
X = my_df.drop(["output"], axis = 1)
y = my_df["output"]

# split into training and  test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

# instantiate our model object

clf = LogisticRegression(random_state = 42)

# train the model
clf.fit(X_train, y_train)

# assess model accuracy

y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)

y_pred_prob = clf.predict_proba(X_test)

# confusion matrix

conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

plt.style.use("seaborn-v0_8-poster")
plt.matshow(conf_matrix, cmap = "coolwarm")
plt.gca().xaxis.tick_bottom()
plt.title("Confusion Matrix")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
for (i, j), corr_value in np.ndenumerate(conf_matrix):
    plt.text(j, i, corr_value, ha = "center", va = "center", fontsize = 20)
plt.show()