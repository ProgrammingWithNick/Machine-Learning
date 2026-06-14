# Onehotencoder for mapping
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

df=pd.read_csv(r"C:\Users\nickk\Downloads\ml\Lab-01\tempbook.csv")
print(df)

x=df[['color','size','price']].values
color_le=LabelEncoder()
x[:,0]=color_le.fit_transform(x[:,0])
print(x)


#One way with onehotencoder for creating dummy features with sklearn
ct = ColumnTransformer([("color", OneHotEncoder(), [0])], remainder = 'passthrough')
x = ct.fit_transform(x)
print(x)

#Other way with pandas
print(pd.get_dummies(df[['color','price']]))
