import pandas as pd
import numpy as np
from sklearn.impute import KNNImporter

my_df = pd.DataFrame()({
    "A": [1,2,3,4,5],
    "B": [1,1,3,3,4],
    "C": [1,2,9,np.nan,20]
    })

knn_imputer = KNNImporter()
knn_imputer = KNNImporter(n_neighbors = 1)
knn_imputer = KNNImporter(n_neighbors = 2)
knn_imputer = KNNImporter(n_neighbors = 2, weights = "distance")
knn_imputer.fit_transform(my_df)

my_df1 = pd.DataFrame(knn_imputer.fit_transform(my_df), columns = my_df.columns)
