# Feature and Class label mapping
import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\nickk\Downloads\ml\Lab-01\tempbook.csv")
print(df)

# feature mapping with numerical value
size_mapping = {'XL': 3, 'L': 2, 'M': 1}
df['size'] = df['size'].map(size_mapping)
print(df)

# reverse mapping of size
inv_size_mapping = {v: k for k, v in size_mapping.items()}
df['size'] = df['size'].map(inv_size_mapping)
print(df)

# class label mapping
class_mapping = {label: idx for idx, label in enumerate(np.unique(df['class']))}
print(class_mapping)

df['class'] = df['class'].map(class_mapping)
print(df)

# reverse mapping of class
inv_class_mapping = {v: k for k, v in class_mapping.items()}
df['class'] = df['class'].map(inv_class_mapping)
print(df)

# with the help of LabelEncoder
from sklearn.preprocessing import LabelEncoder

class_le = LabelEncoder()
y = class_le.fit_transform(df['class'].values)
print(y)

# reverse
print(class_le.inverse_transform(y))