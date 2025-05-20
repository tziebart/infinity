
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd

# import sample data

my_df = pd.read_csv("data/sample_data_regression.csv")

# split into input and output objects.

X = my_df.drop(["output"], axis = 1)
y = my_df["output"]

# Instantiate our GridSearch object

gscv = GridSearchCV(
    estimator = RandomForestRegressor(random_state=42),  
    param_grid= {"n_estimators": [10, 50, 100, 500],
                 "max_depth" : [1,2,3,4,5,6,7,8,9,10,None]},
    cv= 5,
    scoring= "r2",
    n_jobs= -1
    )

# fit to data

gscv.fit(X, y)

gscv.best_score_

# optimal parameters

gscv.best_params_

# create optimal model object

regressor = gscv.best_estimator_