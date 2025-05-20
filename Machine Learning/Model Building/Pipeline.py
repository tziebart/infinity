import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# import sample data

my_df = pd.read_csv("data/pipeline_data.csv")

# split into input and output objects.

X = my_df.drop(["purchase"], axis = 1)
y = my_df["purchase"]

# split into training and  test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

# specify numeric and categorical features

numeric_features = ["age", "credit_score"]
categorical_features = ["gender"]

# setup pipline

# numeric feature transformer

numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer()),
                                      ("scaler", StandardScaler())])

# categorical feature transformer

categorical_transformer = Pipeline(steps= [("imputer", SimpleImputer(strategy= "constant", fill_value = "U")),
                                           ("ohe", OneHotEncoder(handle_unknown= "ignore"))])

# preprocessing Pipeline

preprocessing_pipeline = ColumnTransformer(transformers = [("numeric", numeric_transformer, numeric_features),
                                                           ("categorical", categorical_transformer, categorical_features)])

# apply the pipeling

# logistic regression

clf = Pipeline(steps = [("preprocessing_pipeline", preprocessing_pipeline),
                        ("classifier", LogisticRegression(random_state= 42))])

clf.fit(X_train, y_train)
y_pred_class = clf.predict(X_test)
accuracy_score(y_test, y_pred_class)

# Random forest

clf = Pipeline(steps = [("preprocessing_pipeline", preprocessing_pipeline),
                        ("classifier", RandomForestClassifier(random_state= 42))])

clf.fit(X_train, y_train)
y_pred_class = clf.predict(X_test)
accuracy_score(y_test, y_pred_class)

# save the pipeline

import joblib

joblib.dump(clf, "data/model.joblib")


# import pipeline

import joblib
import pandas as pd
import numpy as np

# import pipeline
clf = joblib.load("data/model.joblib")

# create new data

new_data = pd.DataFrame({"age": [25, np.nan, 50],
                         "gender": ["M", "F", np.nan],
                         "credit_score": [200, 100, 500]})

clf.predict(new_data)
