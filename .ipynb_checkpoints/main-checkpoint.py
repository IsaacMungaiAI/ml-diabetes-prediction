# machine learning model for diabetes prediction

import pandas as pd

df=pd.read_csv('diabetes.csv')
print(df.head())
print(df.isnull().sum())