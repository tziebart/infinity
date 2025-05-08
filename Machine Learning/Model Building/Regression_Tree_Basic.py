# -*- coding: utf-8 -*-

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd

# import sample data
my_df = pd.read_csv("data/sample_data_regression.csv")

# split into input and output objects.
X = my_df.drop(["output"], axis = 1)
y = my_df["output"]

# split into training and  test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# instantiate our model object

regressor = DecisionTreeRegressor(min_samples_leaf=7)

# train the model
regressor.fit(X_train, y_train)

# assess model accuracy

y_pred = regressor.predict(X_test)
r2_score(y_test, y_pred)

# a demonstraton of overfitting
y_pred_training = regressor.predict(X_train)
r2_score(y_train, y_pred_training)

# plot our tree
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(25,15))
tree = plot_tree(regressor, 
                 feature_names = list(X.columns),
                 filled=True,
                 rounded=True,
                 fontsize=21)