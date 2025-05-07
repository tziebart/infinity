# -*- coding: utf-8 -*-
import pandas as pd

my_df = pd.read_csv("feature_selection_sample_data.csv")

# test train split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

X = my_df.drop(["output"], axis = 1)
y = my_df["output"]

# regression model

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
r2_score(y_test, y_pred)

# classification model

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify=y)

# cross validation
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold

cv_scores = cross_val_score(regressor, X, y, cv = 4, scoring="r2")

cv_scores.mean()

# regression
cv = KFold(n_splits=4, shuffle=True, random_state=42)
cv_scores = cross_val_score(regressor, X, y, cv = cv, scoring="r2")
cv_scores.mean()

# classification
cv = KFold(n_splits=4, shuffle=True, random_state=42)
cv_scores = cross_val_score(clf, X, y, cv = cv, scoring="accuracy")
cv_scores.mean()
