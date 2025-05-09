
from sklearn.ensemble import RandomForestRegressor
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

regressor = RandomForestRegressor(random_state=42, n_estimators=1000)

# train the model
regressor.fit(X_train, y_train)

# assess model accuracy

y_pred = regressor.predict(X_test)
r2_score(y_test, y_pred)

# feature importance

regressor.feature_importances_

feature_importance = pd.DataFrame(regressor.feature_importances_)
feature_names = pd.DataFrame(X.columns)
feature_importance_summary = pd.concat([feature_names, feature_importance], axis = 1)
feature_importance_summary.columns = ["input_variable", "feature_importance"]
feature_importance_summary.sort_values(by= "feature_importance", inplace = True)

import matplotlib.pyplot as plt

plt.barh(feature_importance_summary["input_variable"], feature_importance_summary["feature_importance"])
plt.title("Feature Importance of Random Forest")
plt.xlabel("Feature Importance")
plt.tight_layout()
plt.show()
