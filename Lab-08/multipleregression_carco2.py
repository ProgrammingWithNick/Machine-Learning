from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
import numpy as np

df = pd.read_csv("D:\A4087\Lab-08\cardata.csv")

X = df[['Weight', 'Volume']]
y = df['CO2']

X_train,X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 100)

reg_model = linear_model.LinearRegression()

reg_model = LinearRegression().fit(X_train, y_train)

#Printing the model coefficients
print('Intercept: ',reg_model.intercept_)

y_pred= reg_model.predict(X_test)  
x_pred= reg_model.predict(X_train)

print("Prediction for test set: {}".format(y_pred))

reg_model_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred})

mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
r2 = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

print('Mean Absolute Error:', mae)
print('Mean Square Error:', mse)
print('Root Mean Square Error:', r2)

predictedCO2 = reg_model.predict([[2300, 1300]])

print(predictedCO2)
