# -*- coding: utf-8 -*-
import pandas as pd

my_df = pd.read_csv("feature_selection_sample_data.csv")

correlation_matrix = my_df.corr()
