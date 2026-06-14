import numpy as np
import pandas as pd
from io import StringIO
from sklearn.impute import SimpleImputer

narr=np.array([[1,2,3,4],[np.nan,6,7,8],[9,10,np.nan,12]])
print(narr)

df = pd.DataFrame(narr, columns=['A','B','C','D'])

print('---------imputed data with mean---------')
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

#The fit method is used to compute the imputation strategy based #on the provided data X

imputer=imputer.fit(df)

#The transform method is used to apply the imputation strategy #learned during the fit phase to fill in the missing values in #the input data X.

imputed_data=imputer.transform(df)
print(imputed_data)

print('---------imputed data with median---------')
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
imputer=imputer.fit(df)
imputed_data=imputer.transform(df)
print(imputed_data)


print('---------imputed data with constant---------')
imputer = SimpleImputer(missing_values=np.nan, strategy='constant',fill_value=0)
imputer=imputer.fit(df)
imputed_data=imputer.transform(df)
print(imputed_data)



