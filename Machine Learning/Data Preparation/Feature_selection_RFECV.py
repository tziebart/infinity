# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt

my_df = pd.read_csv("feature_selection_sample_data.csv")

from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression

X = my_df.drop(["output"], axis = 1)
y = my_df["output"]

regressor = LinearRegression()
feature_selector = RFECV(regressor)

fit = feature_selector.fit(X, y)

optimal_feature_count = feature_selector.n_features_
print(f"Optimal number of features: {optimal_feature_count}")

X_new = X.loc[:, feature_selector.get_support()]

# Use cv_results_['mean_test_score'] instead of grid_scores_
plt.plot(range(1, len(fit.cv_results_['mean_test_score']) + 1), fit.cv_results_['mean_test_score'], marker="o")
plt.ylabel("Model Score")
plt.xlabel("Number of Features")
plt.title(f"Feature Selection using RFE \n Optimal number of features is {optimal_feature_count}")
plt.tight_layout()
plt.show()