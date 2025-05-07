import pandas as pd
import numpy as np

my_df = pd.DataFrame({"A" : [1,2,4,np.nan, 5, np.nan, 7],
                      "B": [4, np.nan, 7, np.nan, 1, np.nan,2]})

# finding missing values
my_df.isna()
my_df.isna().sum()

# Dropping missing values with pandas

my_df.dropna()
my_df.dropna(how = "any")

my_df.dropna(how = "any", subset = ["A"])

# Filling missing values with pandas

my_df = pd.DataFrame({"A" : [1,2,4,np.nan, 5, np.nan, 7],
                      "B": [4, np.nan, 7, np.nan, 1, np.nan,2]})

my_df.fillna(value = 100)

mean_value = my_df["A"].mean()
my_df["A"].fillna(value = mean_value)

my_df.fillna(value = my_df.mean(), inplace=True)